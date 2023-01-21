import glob
import os


# Convert the .cl files to .h files
cl_files = glob.glob("src/kernels/*.cl")
for cl_file in cl_files:
    fname, _ = os.path.splitext(cl_file)
    os.system(f"xxd -i {cl_file} {fname}.h")
    print(f"Converted {cl_file:35s} --> \t {fname}.h")


# Generate a kernel.h file to include in main.c
names = [os.path.splitext(os.path.basename(x))[0] for x in cl_files]

# Move the declarations file to beginning of the names list
names.remove("declarations")
names.insert(0, "declarations")

f = open("src/kernels/kernels.h", "w")

f.write('#ifndef __KERNELS_H__\n')
f.write('#define __KERNELS_H__\n')
f.write('\n')
f.write('\n')
f.write('#include <stdint.h>\n')
f.write('#include <stdlib.h>\n')
f.write('\n')
f.write('\n')
for name in names:
    f.write(f'#include "{name}.h"\n')    
f.write('\n')
f.write('\n')
f.write('static inline void\n')
f.write('load_kernels(cl_uint * num_sources, char *** sourcestrs, size_t ** sourcelengths)\n')
f.write('{\n')
f.write(f'    *num_sources = {len(names)};\n')
f.write(f'    *sourcestrs = (char **) malloc(sizeof(char *) * {len(names)});\n')
f.write(f'    *sourcelengths = (size_t *) malloc(sizeof(size_t) * {len(names)});\n')
f.write('\n')
for i in range(len(names)):
    f.write(f'    (*sourcestrs)[{i}] = (char *) src_kernels_{names[i]}_cl;\n')
    f.write(f'    (*sourcelengths)[{i}] = src_kernels_{names[i]}_cl_len;\n')
    f.write('\n')
f.write('}\n')
f.write('\n')
f.write('\n')
f.write('#endif // __KERNELS_H__\n')

f.close()
