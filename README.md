# NandFsrReader

FSR BML/STL NAND extractor. Python 3.10+, no dependencies.

## Usage

Export all partitions:

```bash
python3 NandFsrReader.py --nand onenand.bin --oob onenand.oob ./OUTPUT
```

List partitions:

```bash
python3 NandFsrReader.py --list
```

Export single STL partition:

```bash
python3 NandFsrReader.py 0014 partition_0014.bin
```

Export split Flex SLC/MLC dumps:

```bash
python3 NandFsrReader.py --nand onenand_mlc.bin --oob onenand_mlc.oob --slc-nand onenand_slc.bin --slc-oob onenand_slc.oob ./OUTPUT
```

Default inputs: `onenand.bin`, `onenand.oob`. Profiles: `auto`, `512M`, `2G`, `flex`.
