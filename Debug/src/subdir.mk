################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/RRT.cu \
../src/floyd_warshall_algo.cu \
../src/main.cu 

CU_DEPS += \
./src/RRT.d \
./src/floyd_warshall_algo.d \
./src/main.d 

OBJS += \
./src/RRT.o \
./src/floyd_warshall_algo.o \
./src/main.o 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.5/bin/nvcc -I"/usr/local/cuda-6.5/samples/0_Simple" -I"/usr/local/cuda-6.5/samples/common/inc" -I"/home/thenightling/cuda-workspace/RRT" -G -g -O0 -ccbin arm-linux-gnueabihf-g++ -gencode arch=compute_30,code=sm_30 -gencode arch=compute_32,code=sm_32 --target-cpu-architecture ARM -m32 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.5/bin/nvcc -I"/usr/local/cuda-6.5/samples/0_Simple" -I"/usr/local/cuda-6.5/samples/common/inc" -I"/home/thenightling/cuda-workspace/RRT" -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_30,code=compute_30 -gencode arch=compute_32,code=sm_32 --target-cpu-architecture ARM -m32 -ccbin arm-linux-gnueabihf-g++  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


