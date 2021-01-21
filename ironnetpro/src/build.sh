#nvcc -c kernels.cu --compiler-options '-fPIC' -O 3 -arch sm_75 --use_fast_math -D_FORCE_INLINES
nvcc -c kernels.cu --compiler-options '-fPIC -Wall' -O 3 -arch sm_52 --use_fast_math -D_FORCE_INLINES
#nvcc -c kernels.cu --compiler-options '-fPIC -Wall' -O 3 --use_fast_math -D_FORCE_INLINES
gcc -o libkernels.so kernels.o -shared -O3 -march=native -Wall
cp libkernels.so ~/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/lib
cp libkernels.so /home/tapa/.rustup/toolchains/1.46.0-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/lib
