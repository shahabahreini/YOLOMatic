import mmap
import struct
import sys


def clear_execstack(filename):
    with open(filename, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0)
        e_ident = mm[:16]
        if e_ident[:4] != b"\x7fELF":
            print("Not an ELF file")
            return

        is_64 = e_ident[4] == 2
        endian = "<" if e_ident[5] == 1 else ">"

        if is_64:
            e_phoff = struct.unpack(endian + "Q", mm[32:40])[0]
            e_phentsize = struct.unpack(endian + "H", mm[54:56])[0]
            e_phnum = struct.unpack(endian + "H", mm[56:58])[0]
        else:
            e_phoff = struct.unpack(endian + "I", mm[28:32])[0]
            e_phentsize = struct.unpack(endian + "H", mm[42:44])[0]
            e_phnum = struct.unpack(endian + "H", mm[44:46])[0]

        for i in range(e_phnum):
            offset = e_phoff + i * e_phentsize
            if is_64:
                p_type = struct.unpack(endian + "I", mm[offset : offset + 4])[0]
                if p_type == 0x6474E551:  # PT_GNU_STACK
                    p_flags = struct.unpack(endian + "I", mm[offset + 4 : offset + 8])[
                        0
                    ]
                    p_flags &= ~1
                    mm[offset + 4 : offset + 8] = struct.pack(endian + "I", p_flags)
                    print("Cleared PT_GNU_STACK executable flag")
                    mm.flush()
                    return
            else:
                p_type = struct.unpack(endian + "I", mm[offset : offset + 4])[0]
                if p_type == 0x6474E551:
                    p_flags = struct.unpack(
                        endian + "I", mm[offset + 24 : offset + 28]
                    )[0]
                    p_flags &= ~1
                    mm[offset + 24 : offset + 28] = struct.pack(endian + "I", p_flags)
                    print("Cleared PT_GNU_STACK executable flag")
                    mm.flush()
                    return
        print("PT_GNU_STACK not found")
        mm.close()


if __name__ == "__main__":
    clear_execstack(sys.argv[1])
