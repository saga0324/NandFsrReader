import struct
import unittest
import zlib
from types import SimpleNamespace

import NandFsrReader as r


def data_page(logical, kind, value=0):
    data = bytes([value]) * r.PAGE
    spare = bytearray(b'\xff' * r.SPARE)
    struct.pack_into('<HH', spare, 32, (kind << 8 | 0x10) ^ 0xffff, kind << 8 | 0x10)
    if kind in (1, 2):
        struct.pack_into('<H', spare, 4, logical ^ 0xffff)
        struct.pack_into('<H', spare, 16, logical)
    else:
        struct.pack_into('<HH', spare, 18, logical ^ 0xffff, logical)
    for sector, (low, high) in enumerate(r.CRC_HALVES):
        crc = zlib.crc32(data[sector * 512:(sector + 1) * 512])
        struct.pack_into('<H', spare, low, crc & 0xffff)
        struct.pack_into('<H', spare, high, crc >> 16)
    return data, bytes(spare)


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.ftl = r.FTL.__new__(r.FTL)
        self.ftl.pages_per_unit = 128
        self.ftl.nand = SimpleNamespace(geometry=r.Geometry('flex', 1, 2048, 128, 1))

    def test_all_data_tag_types_and_address_inverses(self):
        for kind in (1, 2, 4, 8):
            with self.subTest(kind=kind):
                _, spare = data_page(53, kind)
                self.assertEqual(self.ftl._data_tag(spare, 0), (53, kind))
                broken = bytearray(spare)
                broken[4 if kind in (1, 2) else 18] ^= 1
                with self.assertRaises(r.FormatError):
                    self.ftl._data_tag(broken, 0)

    def test_flex_physical_addresses_and_boundary(self):
        nand = r.Nand.__new__(r.Nand)
        nand.geometry = self.ftl.nand.geometry
        nand.remaps = {}
        self.assertEqual(nand.unit_pages(0), 64)
        self.assertEqual(nand.unit_pages(1), 128)
        self.assertEqual(nand.page_address(0, 63), 63)
        self.assertEqual(nand.page_address(1, 0), 64)
        self.assertEqual(nand.page_address(2047, 127), nand.geometry.size // r.PAGE - 1)
        with self.assertRaises(r.FormatError):
            nand.page_address(0, 64)
        nand.remaps[512] = 2043
        self.assertEqual(nand.page_address(512, 127), nand.physical_page(2043, 127))

    def test_split_image_crosses_slc_mlc_boundary(self):
        image = r.SplitImage(b'abcd', b'efgh')
        self.assertEqual(image[2:6], b'cdef')
        self.assertEqual(image[4:8], b'efgh')

    def replay(self, pages):
        record = bytearray(b'\xff' * 488)
        struct.pack_into('<HHH', record, 20, 10, 52, 9)
        struct.pack_into('<H', record, 224 + 52 * 2, 6)
        self.ftl._read = lambda unit, page: pages.get(page, (r.ERASED, b'\xff' * r.SPARE))
        return self.ftl._recover_log(record, 0, 0, (), 224)

    def test_replay_new_confirmed_upper_page(self):
        pages = {p: data_page(52, 1, p) for p in range(3, 9)}
        pages[9] = data_page(52, 8, 9)
        updated = self.replay(pages)
        self.assertEqual(r.u16(updated, 224 + 52 * 2), 9)

    def test_unconfirmed_tail_does_not_replace_checkpoint(self):
        pages = {p: data_page(52, 1, p) for p in range(3, 10)}
        pages[6] = data_page(52, 2, 6)
        updated = self.replay(pages)
        self.assertEqual(r.u16(updated, 224 + 52 * 2), 6)

    def test_replay_rejects_crc_damage(self):
        pages = {p: data_page(52, 1, p) for p in range(3, 9)}
        data, spare = data_page(52, 8, 9)
        pages[9] = bytes([data[0] ^ 1]) + data[1:], spare
        with self.assertRaisesRegex(r.FormatError, 'CRC mismatch'):
            self.replay(pages)

    def test_existing_dual_die_addresses(self):
        nand = r.Nand.__new__(r.Nand)
        nand.geometry = r.Geometry('2G', 2, 4096)
        nand.remaps = {}
        self.assertEqual(nand.page_address(5, 2), 5 * 64 + 1)
        self.assertEqual(nand.page_address(5, 3), (4096 + 5) * 64 + 1)


if __name__ == '__main__':
    unittest.main()
