#define CL_TARGET_OPENCL_VERSION 300

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <CL/cl.h>

int is_prime(int n)
{
    if(n < 2) return 0;

    for(int i = 2; i * i <= n; i++)
        if(n % i == 0)
            return 0;

    return 1;
}

const char *kernel_src =
"__kernel void prime(__global int *out, int batas){"
" int id=get_global_id(0)+2;"
" if(id>batas) return;"
" int p=1;"
" for(int i=2;i*i<=id;i++){"
"   if(id%i==0){"
"     p=0;"
"     break;"
"   }"
" }"
" out[id-2]=p;"
"}";

int main()
{
    int batas;

    printf("Masukkan batas angka bilangan prima: ");
    scanf("%d",&batas);

    double start = omp_get_wtime();

    int seq_count = 0;

    for(int i=2;i<=batas;i++)
        if(is_prime(i))
            seq_count++;

    double seq_time = omp_get_wtime() - start;

    start = omp_get_wtime();

    int omp_count = 0;

    #pragma omp parallel for reduction(+:omp_count)
    for(int i=2;i<=batas;i++)
        if(is_prime(i))
            omp_count++;

    double omp_time = omp_get_wtime() - start;

    cl_int err;
    cl_platform_id platform;
    cl_device_id device;

    clGetPlatformIDs(1,&platform,NULL);
    clGetDeviceIDs(platform,CL_DEVICE_TYPE_ALL,1,&device,NULL);

    cl_context context =
        clCreateContext(NULL,1,&device,NULL,NULL,&err);

    cl_command_queue queue =
        clCreateCommandQueueWithProperties(
            context,
            device,
            0,
            &err);

    cl_program program =
        clCreateProgramWithSource(
            context,
            1,
            &kernel_src,
            NULL,
            &err);

    err = clBuildProgram(
        program,
        1,
        &device,
        NULL,
        NULL,
        NULL);

    if(err != CL_SUCCESS)
    {
        size_t log_size;

        clGetProgramBuildInfo(
            program,
            device,
            CL_PROGRAM_BUILD_LOG,
            0,
            NULL,
            &log_size);

        char *log = malloc(log_size);

        clGetProgramBuildInfo(
            program,
            device,
            CL_PROGRAM_BUILD_LOG,
            log_size,
            log,
            NULL);

        printf("%s\n",log);

        free(log);
        return 1;
    }

    int count = batas - 1;

    cl_mem buffer =
        clCreateBuffer(
            context,
            CL_MEM_WRITE_ONLY,
            count * sizeof(int),
            NULL,
            &err);

    cl_kernel kernel =
        clCreateKernel(
            program,
            "prime",
            &err);

    clSetKernelArg(kernel,0,sizeof(cl_mem),&buffer);
    clSetKernelArg(kernel,1,sizeof(int),&batas);

    size_t local = 256;
    size_t global =
        ((count + local - 1) / local) * local;

    int *result =
        malloc(count * sizeof(int));

    start = omp_get_wtime();

    clEnqueueNDRangeKernel(
        queue,
        kernel,
        1,
        NULL,
        &global,
        &local,
        0,
        NULL,
        NULL);

    clFinish(queue);

    double ocl_time = omp_get_wtime() - start;

    clEnqueueReadBuffer(
        queue,
        buffer,
        CL_TRUE,
        0,
        count * sizeof(int),
        result,
        0,
        NULL,
        NULL);

    int ocl_count = 0;

    for(int i=0;i<count;i++)
        ocl_count += result[i];

    printf("\nSequential : %.9f detik | prime=%d\n",
           seq_time,
           seq_count);

    printf("OpenMP     : %.9f detik | prime=%d\n",
           omp_time,
           omp_count);

    printf("OpenCL     : %.9f detik | prime=%d\n",
           ocl_time,
           ocl_count);

    if(omp_time > 0)
        printf("\nSpeedup OpenMP : %.3fx\n",
               seq_time / omp_time);

    if(ocl_time > 0)
        printf("Speedup OpenCL : %.3fx\n",
               seq_time / ocl_time);

    free(result);

    clReleaseMemObject(buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}


