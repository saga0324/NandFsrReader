#!/usr/bin/env python3
# FSR STL/FTL NAND Reader
# Yuu 2026
import argparse
from array import array
from contextlib import ExitStack
from dataclasses import dataclass
import mmap
from pathlib import Path
import struct
import sys
import zlib

PAGE = 4096
SPARE = 128
PAGES_PER_BLOCK = 64
ERASED = b'\xff' * PAGE
MASK32 = 0xFFFFFFFF
ZERO_BITS = tuple(8 - n.bit_count() for n in range(256))
MLC_LOWER = (0, 1) + tuple(p for first in range(2, 123, 4) for p in (first, first + 1))
MLC_UPPER = tuple(p for p in range(128) if p not in MLC_LOWER)
MLC_PAIRS = dict(zip(MLC_LOWER, MLC_UPPER)) | dict(zip(MLC_UPPER, MLC_LOWER))
CRC_HALVES = ((36, 48), (50, 52), (64, 66), (68, 80),
              (82, 84), (96, 98), (100, 112), (114, 116))


class FormatError(ValueError):
    pass


def u16(data, offset=0):
    return struct.unpack_from('<H', data, offset)[0]


def u32(data, offset=0):
    return struct.unpack_from('<I', data, offset)[0]


def checksum(data, length, offset, label):
    count, inverse = struct.unpack_from('<II', data, offset)
    if count ^ inverse != MASK32 or sum(ZERO_BITS[b] for b in data[:length]) != count:
        raise FormatError(f'{label}: invalid zero-bit count or inverse')


def age(context):
    return u32(context), u32(context, 4)


@dataclass(frozen=True)
class Geometry:
    name: str
    dies: int
    blocks_per_die: int
    pages_per_block: int = 64
    slc_blocks: int = 0

    @property
    def pages_per_unit(self):
        return self.pages_per_block * self.dies

    @property
    def size(self):
        return (self.blocks_per_die * self.pages_per_block
                - self.slc_blocks * (self.pages_per_block - 64)) * self.dies * PAGE


GEOMETRIES = (Geometry('512M', 1, 2048), Geometry('2G', 2, 4096))


@dataclass(frozen=True)
class Partition:
    id: int
    attributes: int
    start: int
    units: int


class SplitImage:
    def __init__(self, slc, mlc):
        self.slc, self.mlc = slc, mlc

    def __getitem__(self, span):
        start, end = span.start, span.stop
        boundary = len(self.slc)
        if start >= boundary:
            return self.mlc[start - boundary:end - boundary]
        if end <= boundary:
            return self.slc[start:end]
        return self.slc[start:] + self.mlc[:end - boundary]


class Nand:
    def __init__(self, nand_path, oob_path, profile='auto', slc_nand=None, slc_oob=None):
        self.stack = ExitStack()
        self.inputs = (Path(nand_path).resolve(), Path(oob_path).resolve())
        try:
            if (slc_nand is None) != (slc_oob is None):
                raise FormatError('Flex requires both --slc-nand and --slc-oob')
            data_file = self.stack.enter_context(Path(nand_path).open('rb'))
            spare_file = self.stack.enter_context(Path(oob_path).open('rb'))
            data_size = data_file.seek(0, 2)
            spare_size = spare_file.seek(0, 2)
            if slc_nand is not None:
                slc_file = self.stack.enter_context(Path(slc_nand).open('rb'))
                slc_spare_file = self.stack.enter_context(Path(slc_oob).open('rb'))
                slc_size = slc_file.seek(0, 2)
                slc_spare_size = slc_spare_file.seek(0, 2)
                if (not slc_size or slc_size % (64 * PAGE) or not data_size
                        or data_size % (128 * PAGE)
                        or slc_spare_size != slc_size // PAGE * SPARE):
                    raise FormatError('invalid split Flex SLC/MLC sizes')
                slc_blocks = slc_size // (64 * PAGE)
                blocks = slc_blocks + data_size // (128 * PAGE)
                if blocks not in (1024, 2048, 4096, 8192):
                    raise FormatError('unsupported single-die Flex block count')
                candidates = ([Geometry('flex', 1, blocks, 128, slc_blocks)]
                              if profile in ('auto', 'flex') else [])
                self.inputs += (Path(slc_nand).resolve(), Path(slc_oob).resolve())
            else:
                candidates = [g for g in GEOMETRIES if g.size == data_size
                              and profile in ('auto', g.name)]
            if len(candidates) != 1:
                raise FormatError(f'unsupported NAND size/profile: {data_size} / {profile}')
            self.geometry = candidates[0]
            if spare_size != data_size // PAGE * SPARE:
                raise FormatError('OOB must contain 128 bytes per 4096-byte data page')
            self.data = self.stack.enter_context(mmap.mmap(data_file.fileno(), 0, access=mmap.ACCESS_READ))
            self.spare = self.stack.enter_context(mmap.mmap(spare_file.fileno(), 0, access=mmap.ACCESS_READ))
            if slc_nand is not None:
                slc_data = self.stack.enter_context(mmap.mmap(slc_file.fileno(), 0, access=mmap.ACCESS_READ))
                slc_spare = self.stack.enter_context(mmap.mmap(slc_spare_file.fileno(), 0, access=mmap.ACCESS_READ))
                self.data = SplitImage(slc_data, self.data)
                self.spare = SplitImage(slc_spare, self.spare)
            self.remaps = {}
            self.partitions = self._bml()
        except BaseException:
            self.stack.close()
            raise

    def close(self):
        self.stack.close()

    def raw(self, block, page):
        if not 0 <= block < self.geometry.blocks_per_die * self.geometry.dies or not 0 <= page < self.block_pages(block):
            raise FormatError(f'physical address outside image: {block:#x}:{page}')
        physical = self.physical_page(block, page)
        return (self.data[physical * PAGE:(physical + 1) * PAGE],
                self.spare[physical * SPARE:(physical + 1) * SPARE])

    def block_pages(self, block):
        return 64 if block < self.geometry.slc_blocks else self.geometry.pages_per_block

    def unit_pages(self, unit):
        return self.block_pages(unit) * self.geometry.dies

    def physical_page(self, block, page):
        return (block * self.geometry.pages_per_block
                - min(block, self.geometry.slc_blocks) * (self.geometry.pages_per_block - 64) + page)

    def page_address(self, unit, page):
        g = self.geometry
        if not 0 <= unit < g.blocks_per_die or not 0 <= page < self.unit_pages(unit):
            raise FormatError(f'virtual unit/page outside image: {unit:#x}:{page}')
        block = unit + (page % g.dies) * g.blocks_per_die
        seen = set()
        while block in self.remaps:
            if block in seen:
                raise FormatError('cyclic BML replacement map')
            seen.add(block)
            block = self.remaps[block]
        if page // g.dies >= self.block_pages(block):
            raise FormatError('BML replacement crosses SLC/MLC geometry')
        return self.physical_page(block, page // g.dies)

    def read(self, unit, page):
        physical = self.page_address(unit, page)
        return (self.data[physical * PAGE:(physical + 1) * PAGE],
                self.spare[physical * SPARE:(physical + 1) * SPARE])

    def _bml(self):
        partition_copies = []
        g = self.geometry
        for die in range(g.dies):
            headers = {b'UPCH': [], b'LPCH': []}
            for block in range((die + 1) * g.blocks_per_die - max(64, g.blocks_per_die // 32),
                               (die + 1) * g.blocks_per_die):
                for page in range(0, self.block_pages(block), 4):
                    data, spare = self.raw(block, page)
                    if data[:4] not in headers:
                        continue
                    if u16(spare) != 0xFFFF or u16(spare, 2) != 0xA5A5:
                        continue
                    commit, commit_spare = self.raw(block, page + 3)
                    if commit != b'\0' * PAGE or u16(commit_spare, 2) != 0xA5A5:
                        continue
                    headers[data[:4]].append((u32(data, 12), block, page, data))
            for signature in headers:
                if not headers[signature]:
                    raise FormatError(f'die {die}: no committed {signature.decode()} metadata')
                candidates = headers[signature]
                newest = max(item[0] for item in candidates)
                selected = [item for item in candidates if item[0] == newest]
                payloads = set()
                for _, block, page, data in selected:
                    table = self.raw(block, page + 1)[0]
                    count = u32(data, 0x20C) if signature == b'LPCH' else 0
                    if count > 31:
                        raise FormatError('BML partition count exceeds table capacity')
                    payloads.add((data[0x200:0x210 + count * 16] if count else b'', table))
                if len(payloads) != 1:
                    raise FormatError(f'die {die}: conflicting BML copies at sequence {newest}')
                _, block, page, data = selected[0]
                table = self.raw(block, page + 1)[0]
                for offset in range(4, PAGE, 4):
                    original, replacement = struct.unpack_from('<HH', table, offset)
                    if (original, replacement) == (0xFFFF, 0xFFFF):
                        continue
                    if not (die * g.blocks_per_die <= original < (die + 1) * g.blocks_per_die
                            and die * g.blocks_per_die <= replacement < (die + 1) * g.blocks_per_die):
                        raise FormatError('BML replacement outside its die')
                    if original in self.remaps and self.remaps[original] != replacement:
                        raise FormatError('conflicting BML replacement entries')
                    self.remaps[original] = replacement
                if signature == b'LPCH':
                    if data[0x200:0x208] != b'FSRPARTI':
                        raise FormatError('missing FSRPARTI table')
                    partitions = []
                    for i in range(u32(data, 0x20C)):
                        entry = Partition(*struct.unpack_from('<4H', data, 0x210 + 16 * i))
                        if not entry.units or entry.start + entry.units > g.blocks_per_die:
                            raise FormatError('invalid BML partition extent')
                        partitions.append(entry)
                    partition_copies.append(partitions)
        if any(copy != partition_copies[0] for copy in partition_copies[1:]):
            raise FormatError('partition tables disagree between dies')
        if len({p.id for p in partition_copies[0]}) != len(partition_copies[0]):
            raise FormatError('duplicate partition IDs')
        return {p.id: p for p in partition_copies[0]}


@dataclass
class Zone:
    index: int
    units: int
    sectors: int
    mapping: array
    pmt_groups: int
    bmt_areas: int


class FTL:
    def __init__(self, nand, partition_id):
        self.nand = nand
        if partition_id not in nand.partitions:
            raise FormatError(f'partition {partition_id:#x} is absent')
        self.partition = nand.partitions[partition_id]
        self.pages_per_unit = nand.unit_pages(self.partition.start)
        if any(nand.unit_pages(unit) != self.pages_per_unit
               for unit in range(self.partition.start, self.partition.start + self.partition.units)):
            raise FormatError('STL partition crosses SLC/MLC boundary')
        self.root = self._root()
        if u32(self.root, 8) != 0x01010100:
            raise FormatError('unsupported STL root format version')
        if u16(self.root, 0x1E) != 4:
            raise FormatError('unsupported STL root-unit count')
        self.zone_count = u16(self.root, 0x1C)
        if self.zone_count not in (1, 2):
            raise FormatError('only one- and two-zone STL partitions are supported')
        if u16(self.root, 0x20) != 1 or u16(self.root, 0x22) != 8:
            raise FormatError('unsupported STL data-group/log geometry')
        self.zones = [self._zone(z) for z in range(self.zone_count)]
        self.total_sectors = sum(zone.sectors for zone in self.zones)
        if self.zone_count > 1 and len({zone.sectors for zone in self.zones}) != 1:
            raise FormatError('unequal interleaved zone sizes are unsupported')

    def _read(self, relative_unit, page):
        if not 0 <= relative_unit < self.partition.units:
            raise FormatError(f'STL unit {relative_unit:#x} outside partition')
        return self.nand.read(self.partition.start + relative_unit, page)

    def _root(self):
        roots = []
        for unit in range(min(4, self.partition.units)):
            for page in range(self.pages_per_unit):
                data, spare = self._read(unit, page)
                if data[:8] != b'FSR_STL\0':
                    continue
                if u16(spare, 32) ^ u16(spare, 34) != 0xFFFF or u16(spare, 34) != 0x0140:
                    raise FormatError('invalid STL root spare tag')
                if u16(spare, 4) != sum(ZERO_BITS[b] for b in data[:0x738]):
                    raise FormatError('STL root zero-bit count mismatch')
                roots.append(((u32(data, 12), u32(data, 16)), data))
        if not roots:
            raise FormatError('no supported STL root in first four units')
        newest = min(key for key, _ in roots)
        copies = {data for key, data in roots if key == newest}
        if len(copies) != 1:
            raise FormatError('conflicting STL root copies')
        return copies.pop()

    def _data_tag(self, spare, index):
        kind = spare[35]
        if (spare[34] != 0x10 + index or kind not in (1, 2, 4, 8)
                or u16(spare, 32) ^ u16(spare, 34) != 0xFFFF):
            raise FormatError('invalid data page tag/inverse')
        if kind in (1, 2):
            logical, inverse, unused = u16(spare, 16), u16(spare, 4), spare[18:22]
        else:
            logical, inverse, unused = u16(spare, 20), u16(spare, 18), spare[4:6] + spare[16:18]
        if logical ^ inverse != 0xFFFF or unused != b'\xff' * 4:
            raise FormatError('invalid data page address/inverse')
        return logical, kind

    def _recover_log(self, record, slot, index, meta_units, table_offset):
        descriptor = 16 + slot * 20
        unit, next_page = u16(record, descriptor + 4), u16(record, descriptor + 8)
        if next_page > self.pages_per_unit:
            raise FormatError('active log checkpoint outside unit')
        if next_page == self.pages_per_unit:
            return record
        if unit in meta_units:
            raise FormatError('active log overlaps metadata')
        start = next_page
        if self.nand.geometry.pages_per_block == 128:
            upper = next((p for p in range(start, 128) if p in MLC_UPPER), None)
            if upper is None:
                return record
            start = MLC_PAIRS[upper]
        pending, confirmed, erased = [], -1, 0
        for page in range(start, self.pages_per_unit):
            data, spare = self._read(unit, page)
            if data == ERASED and spare == b'\xff' * SPARE:
                erased += 1
                if erased > self.nand.geometry.dies:
                    break
                continue
            erased = 0
            logical, kind = self._data_tag(spare, index)
            if logical >= self.pages_per_unit:
                raise FormatError('active log logical page outside group')
            if kind in (2, 8):
                for sector, (low, high) in enumerate(CRC_HALVES):
                    stored = u16(spare, low) | u16(spare, high) << 16
                    if zlib.crc32(data[sector * 512:(sector + 1) * 512]) != stored:
                        raise FormatError(f'active log CRC mismatch at {unit:#x}:{page}')
                confirmed = page
            if page >= next_page:
                pending.append((page, logical))
        result = bytearray(record)
        for page, logical in pending:
            if page <= confirmed:
                struct.pack_into('<H', result, table_offset + logical * 2,
                                 slot * self.pages_per_unit + page)
        return result

    def _zone(self, index):
        info = self.root[0x38 + index * 224:0x38 + (index + 1) * 224]
        meta_count, units, groups_per_page = u16(info, 4), u16(info, 14), u16(info, 10)
        sectors, area_size = u32(info, 16), u16(info, 26)
        if meta_count not in (16, 24, 32) or area_size != 512 or not groups_per_page:
            raise FormatError('unsupported STL zone geometry')
        if not units or sectors != units * self.pages_per_unit * 8:
            raise FormatError('zone sector count disagrees with geometry')
        record_size = 16 + 8 * 20 + 8 * 2 + 8 * 2 + 8 * self.nand.geometry.dies * 2 + self.pages_per_unit * 2 + 8
        table_offset = record_size - 8 - self.pages_per_unit * 2
        if 512 + groups_per_page * record_size > PAGE:
            raise FormatError('PMT records exceed metadata page')
        meta_units = struct.unpack_from(f'<{meta_count}H', info, 32)
        if len(set(meta_units)) != meta_count:
            raise FormatError('duplicate metadata unit numbers')
        pmts, bmts, contexts = {}, {}, {}

        def retain(table, key, version, payload):
            if key in table and table[key][0] == version and table[key][1] != payload:
                raise FormatError(f'zone {index}: conflicting mapping at same age')
            if key not in table or version < table[key][0]:
                table[key] = (version, payload)

        for unit in meta_units:
            for page in range(self.pages_per_unit):
                data, spare = self._read(unit, page)
                tag = u16(spare, 34)
                if spare == b'\xff' * SPARE:
                    continue
                if spare[34] != 0x40 + index:
                    raise FormatError(f'zone {index}: wrong metadata zone tag at {unit:#x}:{page}')
                if u16(spare, 32) ^ tag != 0xFFFF:
                    raise FormatError('metadata spare tag/inverse mismatch')
                if self.nand.geometry.name == 'flex' and spare[35] in (4, 8):
                    context = data[:512] if spare[35] == 8 else data[-512:]
                    context_length = (300 + 6 * ((units + area_size - 1) // area_size) + 3) & ~3
                    checksum(context, context_length, context_length, 'STL context')
                    retain(contexts, 0, age(context), context[22:24] + context[76:108])
                if spare[35] == 8:
                    version = age(data)
                    for slot in range(groups_per_page):
                        record = data[512 + slot * record_size:512 + (slot + 1) * record_size]
                        group = u16(record)
                        if group == 0xFFFF:
                            continue
                        if group >= units:
                            raise FormatError('PMT data-group number outside zone')
                        checksum(record, record_size - 8, record_size - 8, 'PMT')
                        retain(pmts, group, version, record)
                elif spare[35] == 4:
                    area = u16(data)
                    if area >= (units + area_size - 1) // area_size:
                        raise FormatError('BMT area outside zone')
                    length = (4 + area_size * 6 + area_size // 8 + 3) & ~3
                    checksum(data, length, length, 'BMT')
                    retain(bmts, area, age(data[PAGE - 512:]), data)
                elif spare[35] != 2:
                    raise FormatError(f'unsupported metadata page kind {spare[35]:#x}')
        if len(bmts) != (units + area_size - 1) // area_size:
            raise FormatError(f'zone {index}: missing BMT area(s)')
        if self.nand.geometry.name == 'flex':
            if not contexts:
                raise FormatError('missing STL context')
            context = contexts[0][1]
            count = u16(context)
            if count > 8:
                raise FormatError('unsupported active-log count')
            for entry in range(count):
                group, unit = struct.unpack_from('<HH', context, 2 + entry * 4)
                if group not in pmts:
                    raise FormatError('active log has no PMT record')
                version, record = pmts[group]
                slots = [slot for slot in range(8) if u16(record, 20 + slot * 20) == unit]
                if len(slots) != 1:
                    raise FormatError('active log missing or duplicated in PMT')
                pmts[group] = version, self._recover_log(record, slots[0], index, meta_units, table_offset)
        mapping = array('i', [-1]) * (units * self.pages_per_unit)
        for group in range(units):
            bmt = bmts[group // area_size][1]
            entry = group % area_size
            base = u16(bmt, 4 + entry * 2)
            deleted = bmt[4 + area_size * 6 + entry // 8] & (1 << (entry % 8))
            record = pmts[group][1] if group in pmts else None
            for logical in range(self.pages_per_unit):
                unit, page = (None, None) if base == 0xFFFF or deleted else (base, logical)
                if record is not None:
                    packed = u16(record, table_offset + 2 * logical)
                    if packed != 0xFFFF:
                        log, page = divmod(packed, self.pages_per_unit)
                        if log >= 8:
                            raise FormatError('PMT log index outside log table')
                        unit = u16(record, 16 + log * 20 + 4)
                if unit is not None:
                    if not 0 <= unit < self.partition.units or unit in meta_units:
                        raise FormatError('data mapping overlaps metadata or leaves partition')
                    physical = self.nand.page_address(self.partition.start + unit, page)
                    spare = self.nand.spare[physical * SPARE:(physical + 1) * SPARE]
                    if self._data_tag(spare, index)[0] != logical:
                        raise FormatError(f'data page tag/address mismatch in zone {index}, group {group}, page {logical}')
                    mapping[group * self.pages_per_unit + logical] = physical
        return Zone(index, units, sectors, mapping, len(pmts), len(bmts))

    def logical_page(self, page):
        if not 0 <= page < self.total_sectors // 8:
            raise FormatError('logical page outside partition')
        group, offset = divmod(page, self.pages_per_unit)
        zone = self.zones[group % self.zone_count]
        physical = zone.mapping[(group // self.zone_count) * self.pages_per_unit + offset]
        return ERASED if physical == -1 else self.nand.data[physical * PAGE:(physical + 1) * PAGE]

    def export(self, output, start=0, sectors=None):
        output = Path(output)
        for source in self.nand.inputs:
            if output.resolve() == source or (output.exists() and output.samefile(source)):
                raise FormatError('output must not overwrite a NAND/OOB input')
        end = self.total_sectors if sectors is None else start + sectors
        if not 0 <= start <= end <= self.total_sectors:
            raise FormatError(f'bad sector range; partition has {self.total_sectors} sectors')
        with Path(output).open('wb') as stream:
            position = start
            while position < end:
                page, offset = divmod(position, 8)
                count = min(8 - offset, end - position)
                stream.write(self.logical_page(page)[offset * 512:(offset + count) * 512])
                position += count
        return (end - start) * 512


def integer(value):
    result = int(value, 0)
    if result < 0:
        raise argparse.ArgumentTypeError('must be nonnegative')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('targets', nargs='*', metavar='PATH',
                        help='output directory, or hexadecimal partition ID followed by output file')
    parser.add_argument('--nand', type=Path, default=Path('onenand.bin'))
    parser.add_argument('--oob', type=Path, default=Path('onenand.oob'))
    parser.add_argument('--slc-nand', type=Path, help='SLC data for split Flex dumps (--nand supplies MLC)')
    parser.add_argument('--slc-oob', type=Path, help='SLC spare data for split Flex dumps')
    parser.add_argument('--profile', choices=('auto', '512M', '2G', 'flex'), default='auto')
    parser.add_argument('--list', action='store_true', help='list BML partitions and replacement blocks')
    parser.add_argument('--start-sector', type=integer, default=0)
    parser.add_argument('--sectors', type=integer)
    args = parser.parse_args(argv)
    args.partition = None
    args.output = None
    if len(args.targets) == 1:
        args.output = Path(args.targets[0])
    elif len(args.targets) == 2:
        try:
            args.partition = int(args.targets[0], 16)
        except ValueError:
            parser.error('partition ID must be hexadecimal')
        args.output = Path(args.targets[1])
    elif args.targets or not args.list:
        parser.error('provide an output directory, or a partition ID and output file')
    if not args.list and args.partition is None and (args.start_sector or args.sectors is not None):
        parser.error('sector ranges require a partition ID and output file')
    if args.output is not None:
        for source in (args.nand, args.oob, args.slc_nand, args.slc_oob):
            if source is None:
                continue
            if args.output.resolve() == source.resolve() or (
                    args.output.exists() and args.output.samefile(source)):
                raise FormatError('output must not overwrite a NAND/OOB input')
    nand = Nand(args.nand, args.oob, args.profile, args.slc_nand, args.slc_oob)
    try:
        print(f'profile={nand.geometry.name}; BML replacements={len(nand.remaps)}')
        if args.list:
            print('\nBML partitions:')
            print(f'{"ID":>6}  {"Attributes":>10}  {"Start unit":>10}  {"Units":>6}  {"Size (bytes)":>14}')
            for partition in nand.partitions.values():
                size = sum(nand.unit_pages(unit) for unit in range(
                    partition.start, partition.start + partition.units)) * PAGE
                print(f'{partition.id:#06x}  {partition.attributes:#010x}  '
                      f'{partition.start:#010x}  {partition.units:6d}  {size:14d}')
            print('\nBML replacement blocks (physical):')
            if nand.remaps:
                print(f'{"Original":>10}  {"Replacement":>11}')
                for original, replacement in sorted(nand.remaps.items()):
                    print(f'{original:#010x}  {replacement:#010x}')
            else:
                print('  None')
            return
        if args.partition is None:
            args.output.mkdir(parents=True, exist_ok=True)
            for partition in nand.partitions.values():
                output = args.output / f'partition_{partition.id:04x}.bin'
                for source in nand.inputs:
                    if output.resolve() == source or (output.exists() and output.samefile(source)):
                        raise FormatError('output must not overwrite a NAND/OOB input')
                has_stl = any(
                    nand.read(partition.start + unit, page)[0][:8] == b'FSR_STL\0'
                    for unit in range(min(4, partition.units))
                    for page in range(nand.unit_pages(partition.start + unit))
                )
                print(f'partition {partition.id:#06x}: {"STL" if has_stl else "BML"}')
                if has_stl:
                    ftl = FTL(nand, partition.id)
                    for zone in ftl.zones:
                        print(f'zone {zone.index}: {zone.sectors} sectors, {zone.pmt_groups} PMT groups, {zone.bmt_areas} BMT areas')
                    size = ftl.export(output)
                else:
                    with output.open('wb') as stream:
                        for unit in range(partition.start, partition.start + partition.units):
                            for page in range(nand.unit_pages(unit)):
                                stream.write(nand.read(unit, page)[0])
                    size = output.stat().st_size
                print(f'saved {output} ({size} bytes)')
            return
        ftl = FTL(nand, args.partition)
        for zone in ftl.zones:
            print(f'zone {zone.index}: {zone.sectors} sectors, {zone.pmt_groups} PMT groups, {zone.bmt_areas} BMT areas')
        size = ftl.export(args.output, args.start_sector, args.sectors)
        print(f'saved {args.output} ({size} bytes)')
    finally:
        nand.close()


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, struct.error) as error:
        print(f'error: {error}', file=sys.stderr)
        sys.exit(1)
