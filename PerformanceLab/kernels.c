/********************************************************
 * Kernels to be optimized for the OS&C prflab.
 * Acknowledgment: This lab is an extended version of the
 * CS:APP Performance Lab
 ********************************************************/

#include <pthread.h>
#include <limits.h>
#include <immintrin.h>
#include <string.h>
#include <stdio.h>
#include "defs.h"
#include "smooth.h" // helper functions for naive_smooth
#include "blend.h"  // helper functions for blend_pixel

/*
 * Please fill in the following struct
 */
student_t student = {
    "Jacob",       /* ITU alias */
    "Jacob Grum",  /* Full name */
    "jacg@itu.dk", /* Email address */
};

/******************************************************************************
 * ROTATE KERNEL
 *****************************************************************************/

// Your different versions of the rotate kernel go here

/*
 * naive_rotate - The naive baseline version of rotate
 */
/* stride pattern, visualization (we recommend that you draw this for your functions):
    dst         src
    3 7 B F     0 1 2 3
    2 6 A E     4 5 6 7
    1 5 9 D     8 9 A B
    0 4 8 C     C D E F
 */
char naive_rotate_descr[] = "naive_rotate: Naive baseline implementation";
void naive_rotate(int dim, pixel *src, pixel *dst)
{
    int i, j;

    for (i = 0; i < dim; i++)
        for (j = 0; j < dim; j++)
            dst[RIDX(dim - 1 - j, i, dim)] = src[RIDX(i, j, dim)];
}

char unrolled32_descr[] = "Loop unrolling 32 per iteration, storing two RIDX calculations, to only recalculate every 32 elements";
void unrolled32(int dim, pixel *src, pixel *dst)
{
    int i, j, dstElement, srcElement;
    for (j = 0; j < dim; j++)
    {
        for (i = 0; i < dim; i = i + 32)
        {
            dstElement = RIDX(dim - 1 - j, i, dim);
            srcElement = RIDX(i, j, dim);
            dst[dstElement] = src[srcElement];
            dst[dstElement + 1] = src[srcElement + dim];
            dst[dstElement + 2] = src[srcElement + dim * 2];
            dst[dstElement + 3] = src[srcElement + dim * 3];
            dst[dstElement + 4] = src[srcElement + dim * 4];
            dst[dstElement + 5] = src[srcElement + dim * 5];
            dst[dstElement + 6] = src[srcElement + dim * 6];
            dst[dstElement + 7] = src[srcElement + dim * 7];
            dst[dstElement + 8] = src[srcElement + dim * 8];
            dst[dstElement + 9] = src[srcElement + dim * 9];
            dst[dstElement + 10] = src[srcElement + dim * 10];
            dst[dstElement + 11] = src[srcElement + dim * 11];
            dst[dstElement + 12] = src[srcElement + dim * 12];
            dst[dstElement + 13] = src[srcElement + dim * 13];
            dst[dstElement + 14] = src[srcElement + dim * 14];
            dst[dstElement + 15] = src[srcElement + dim * 15];
            dst[dstElement + 16] = src[srcElement + dim * 16];
            dst[dstElement + 17] = src[srcElement + dim * 17];
            dst[dstElement + 18] = src[srcElement + dim * 18];
            dst[dstElement + 19] = src[srcElement + dim * 19];
            dst[dstElement + 20] = src[srcElement + dim * 20];
            dst[dstElement + 21] = src[srcElement + dim * 21];
            dst[dstElement + 22] = src[srcElement + dim * 22];
            dst[dstElement + 23] = src[srcElement + dim * 23];
            dst[dstElement + 24] = src[srcElement + dim * 24];
            dst[dstElement + 25] = src[srcElement + dim * 25];
            dst[dstElement + 26] = src[srcElement + dim * 26];
            dst[dstElement + 27] = src[srcElement + dim * 27];
            dst[dstElement + 28] = src[srcElement + dim * 28];
            dst[dstElement + 29] = src[srcElement + dim * 29];
            dst[dstElement + 30] = src[srcElement + dim * 30];
            dst[dstElement + 31] = src[srcElement + dim * 31];
        }
    }
}

char rotate_blocking8_descr[] = "Loop unrolling 8 per iteration with 8x8 blocks. Storing two RIDX calculations, to only recalculate every 8 elements ";
void rotate_blocking8(int dim, pixel *src, pixel *dst)
{
    int i, j, jj, dstElement, srcElement;

    // Create blocks 8x8
    for (j = 0; j < dim; j += 8)
    {
        for (i = 0; i < dim; i += 8)
        {
            // unroll the inner loop with factor of 8.
            for (jj = 0; jj < 8; jj++)
            {
                // Calculate dstElement and srcElement once, to reuse
                dstElement = RIDX(dim - 1 - (j + jj), i, dim);
                srcElement = RIDX(i, j + jj, dim);

                // dstElement increases by 1 every time, and src element increases by dimension every time.
                dst[dstElement] = src[srcElement];
                dst[dstElement + 1] = src[srcElement + dim];
                dst[dstElement + 2] = src[srcElement + dim * 2];
                dst[dstElement + 3] = src[srcElement + dim * 3];
                dst[dstElement + 4] = src[srcElement + dim * 4];
                dst[dstElement + 5] = src[srcElement + dim * 5];
                dst[dstElement + 6] = src[srcElement + dim * 6];
                dst[dstElement + 7] = src[srcElement + dim * 7];
            }
        }
    }
}

char rotate_blocking16_descr[] = "Loop unrolling 16 per iteration with 16x16 blocks. Storing two RIDX calculations, to only recalculate every 16 elements ";
void rotate_blocking16(int dim, pixel *src, pixel *dst)
{
    int i, j, jj, dstElement, srcElement;

    // Create blocks 16x16
    for (j = 0; j < dim; j += 16)
    {
        for (i = 0; i < dim; i += 16)
        {
            // unroll the inner loop with a factor of 16.
            for (jj = 0; jj < 16; jj++)
            {
                // Calculate dstElement and srcElement once, to reuse
                dstElement = RIDX(dim - 1 - (j + jj), i, dim);
                srcElement = RIDX(i, j + jj, dim);

                // dstElement increases by 1 every time, and src element increases by dimension every time.
                dst[dstElement] = src[srcElement];
                dst[dstElement + 1] = src[srcElement + dim];
                dst[dstElement + 2] = src[srcElement + dim * 2];
                dst[dstElement + 3] = src[srcElement + dim * 3];
                dst[dstElement + 4] = src[srcElement + dim * 4];
                dst[dstElement + 5] = src[srcElement + dim * 5];
                dst[dstElement + 6] = src[srcElement + dim * 6];
                dst[dstElement + 7] = src[srcElement + dim * 7];
                dst[dstElement + 8] = src[srcElement + dim * 8];
                dst[dstElement + 9] = src[srcElement + dim * 9];
                dst[dstElement + 10] = src[srcElement + dim * 10];
                dst[dstElement + 11] = src[srcElement + dim * 11];
                dst[dstElement + 12] = src[srcElement + dim * 12];
                dst[dstElement + 13] = src[srcElement + dim * 13];
                dst[dstElement + 14] = src[srcElement + dim * 14];
                dst[dstElement + 15] = src[srcElement + dim * 15];
            }
        }
    }
}

char rotate_blocking32_descr[] = "Loop unrolling 32 per iteration with 32x32 blocks. Storing two RIDX calculations, to only recalculate every 32 elements ";
void rotate_blocking32(int dim, pixel *src, pixel *dst)
{
    int i, j, jj, dstElement, srcElement;

    // Create blocks 32x32
    for (j = 0; j < dim; j += 32)
    {
        for (i = 0; i < dim; i += 32)
        {
            // unroll the inner loop with a factor of 32.
            for (jj = 0; jj < 32; jj++)
            {
                // Calculate dstElement and srcElement once, to reuse
                dstElement = RIDX(dim - 1 - (j + jj), i, dim);
                srcElement = RIDX(i, j + jj, dim);

                // dstElement increases by 1 every time, and src element increases by dimension every time.
                dst[dstElement] = src[srcElement];
                dst[dstElement + 1] = src[srcElement + dim];
                dst[dstElement + 2] = src[srcElement + dim * 2];
                dst[dstElement + 3] = src[srcElement + dim * 3];
                dst[dstElement + 4] = src[srcElement + dim * 4];
                dst[dstElement + 5] = src[srcElement + dim * 5];
                dst[dstElement + 6] = src[srcElement + dim * 6];
                dst[dstElement + 7] = src[srcElement + dim * 7];
                dst[dstElement + 8] = src[srcElement + dim * 8];
                dst[dstElement + 9] = src[srcElement + dim * 9];
                dst[dstElement + 10] = src[srcElement + dim * 10];
                dst[dstElement + 11] = src[srcElement + dim * 11];
                dst[dstElement + 12] = src[srcElement + dim * 12];
                dst[dstElement + 13] = src[srcElement + dim * 13];
                dst[dstElement + 14] = src[srcElement + dim * 14];
                dst[dstElement + 15] = src[srcElement + dim * 15];
                dst[dstElement + 16] = src[srcElement + dim * 16];
                dst[dstElement + 17] = src[srcElement + dim * 17];
                dst[dstElement + 18] = src[srcElement + dim * 18];
                dst[dstElement + 19] = src[srcElement + dim * 19];
                dst[dstElement + 20] = src[srcElement + dim * 20];
                dst[dstElement + 21] = src[srcElement + dim * 21];
                dst[dstElement + 22] = src[srcElement + dim * 22];
                dst[dstElement + 23] = src[srcElement + dim * 23];
                dst[dstElement + 24] = src[srcElement + dim * 24];
                dst[dstElement + 25] = src[srcElement + dim * 25];
                dst[dstElement + 26] = src[srcElement + dim * 26];
                dst[dstElement + 27] = src[srcElement + dim * 27];
                dst[dstElement + 28] = src[srcElement + dim * 28];
                dst[dstElement + 29] = src[srcElement + dim * 29];
                dst[dstElement + 30] = src[srcElement + dim * 30];
                dst[dstElement + 31] = src[srcElement + dim * 31];
            }
        }
    }
}

char rotate_blocking32_pointer_descr[] = "Loop unrolling 32 per iteration with 32x32 blocks. Storing two RIDX calculations, to only recalculate every 32 elements ";
void rotate_blocking32_pointer(int dim, pixel *src, pixel *dst)
{
    int i, j, jj;

    // Create blocks 32x32
    for (j = 0; j < dim; j += 32)
    {
        for (i = 0; i < dim; i += 32)
        {
            // unroll the inner loop with a factor of 32.
            for (jj = 0; jj < 32; jj++)
            {
                // Calculate dstElement and srcElement once, to reuse
                // Using pointers, for faster access / pointer arithmetic.
                pixel *dstElement = dst + RIDX(dim - 1 - (j + jj), i, dim);
                pixel *srcElement = src + RIDX(i, j + jj, dim);

                // dstElement increases by 1 every time, and src element increases by dimension every time.
                dstElement[0] = srcElement[0];
                dstElement[1] = srcElement[dim];
                dstElement[2] = srcElement[dim * 2];
                dstElement[3] = srcElement[dim * 3];
                dstElement[4] = srcElement[dim * 4];
                dstElement[5] = srcElement[dim * 5];
                dstElement[6] = srcElement[dim * 6];
                dstElement[7] = srcElement[dim * 7];
                dstElement[8] = srcElement[dim * 8];
                dstElement[9] = srcElement[dim * 9];
                dstElement[10] = srcElement[dim * 10];
                dstElement[11] = srcElement[dim * 11];
                dstElement[12] = srcElement[dim * 12];
                dstElement[13] = srcElement[dim * 13];
                dstElement[14] = srcElement[dim * 14];
                dstElement[15] = srcElement[dim * 15];
                dstElement[16] = srcElement[dim * 16];
                dstElement[17] = srcElement[dim * 17];
                dstElement[18] = srcElement[dim * 18];
                dstElement[19] = srcElement[dim * 19];
                dstElement[20] = srcElement[dim * 20];
                dstElement[21] = srcElement[dim * 21];
                dstElement[22] = srcElement[dim * 22];
                dstElement[23] = srcElement[dim * 23];
                dstElement[24] = srcElement[dim * 24];
                dstElement[25] = srcElement[dim * 25];
                dstElement[26] = srcElement[dim * 26];
                dstElement[27] = srcElement[dim * 27];
                dstElement[28] = srcElement[dim * 28];
                dstElement[29] = srcElement[dim * 29];
                dstElement[30] = srcElement[dim * 30];
                dstElement[31] = srcElement[dim * 31];
            }
        }
    }
}

/*
 * rotate - Your current working version of rotate
 * IMPORTANT: This is the version you will be graded on
 */
char rotate_descr[] = "rotate: Current working version ROTATE";
void rotate(int dim, pixel *src, pixel *dst)
{
    naive_rotate(dim, src, dst);
}

/*
 * register_rotate_functions - Register all of your different versions
 *     of the rotate kernel with the driver by calling the
 *     add_rotate_function() for each test function.
 */
void register_rotate_functions()
{
    add_rotate_function(&rotate, rotate_descr);
    add_rotate_function(&unrolled32, unrolled32_descr);
    add_rotate_function(&rotate_blocking8, rotate_blocking8_descr);
    add_rotate_function(&rotate_blocking16, rotate_blocking16_descr);
    add_rotate_function(&rotate_blocking32, rotate_blocking32_descr);
    add_rotate_function(&rotate_blocking32_pointer, rotate_blocking32_pointer_descr);
}

/******************************************************************************
 * ROTATE_T KERNEL
 *****************************************************************************/

// Your different versions of the rotate_t kernel go here
// (i.e. rotate with multi-threading)

/*
 * rotate_t - Your current working version of rotate_t
 * IMPORTANT: This is the version you will be graded on
 */
char rotate_t_descr[] = "rotate_t: Current working version ROTATE_T";
void rotate_t(int dim, pixel *src, pixel *dst)
{
    naive_rotate(dim, src, dst);
}

typedef struct
{
    int dim;
    pixel *src;
    pixel *dst;
    int start_row;
    int end_row;
} ThreadData;

void *naive_rotate_thread(void *arg)
{
    ThreadData *data = (ThreadData *)arg;
    int dim = data->dim;
    pixel *src = data->src;
    pixel *dst = data->dst;

    for (int i = data->start_row; i < data->end_row; i++)
        for (int j = 0; j < dim; j++)
            dst[RIDX(dim - 1 - j, i, dim)] = src[RIDX(i, j, dim)];

    pthread_exit(NULL);
}

char pthread_naive_rotate_descr[] = "Naive, but with multithreading";
void pthread_naive_rotate(int dim, pixel *src, pixel *dst)
{
    int NUM_THREADS = 1;
    if (dim < 255)
    {
        rotate_blocking32_pointer(dim, src, dst);
        return;
    }
    else if (dim > 255 && dim < 511)
    {
        NUM_THREADS = 4;
    }
    else if (dim >= 500)
    {
        NUM_THREADS = 8;
    }
    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];

    int rows_per_thread = dim / NUM_THREADS;
    int remaining_rows = dim % NUM_THREADS;
    int current_row = 0;

    for (int i = 0; i < NUM_THREADS; i++)
    {
        thread_data[i].dim = dim;
        thread_data[i].src = src;
        thread_data[i].dst = dst;
        thread_data[i].start_row = current_row;
        thread_data[i].end_row = current_row + rows_per_thread + (i < remaining_rows ? 1 : 0);

        pthread_create(&threads[i], NULL, naive_rotate_thread, (void *)&thread_data[i]);

        current_row = thread_data[i].end_row;
    }

    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);
}

void *rotate_thread_blocking8(void *arg)
{
    ThreadData *data = (ThreadData *)arg;
    int dim = data->dim;
    pixel *src = data->src;
    pixel *dst = data->dst;

    for (int j = data->start_row; j < data->end_row; j += 8)
    {
        for (int i = 0; i < dim; i += 8)
        {
            for (int jj = 0; jj < 8; jj++)
            {
                int dstElement = RIDX(dim - 1 - (j + jj), i, dim);
                int srcElement = RIDX(i, j + jj, dim);

                dst[dstElement] = src[srcElement];
                dst[dstElement + 1] = src[srcElement + dim];
                dst[dstElement + 2] = src[srcElement + dim * 2];
                dst[dstElement + 3] = src[srcElement + dim * 3];
                dst[dstElement + 4] = src[srcElement + dim * 4];
                dst[dstElement + 5] = src[srcElement + dim * 5];
                dst[dstElement + 6] = src[srcElement + dim * 6];
                dst[dstElement + 7] = src[srcElement + dim * 7];
            }
        }
    }

    pthread_exit(NULL);
}

char pthread_rotate_blocking8_descr[] = "Pthread implementation with loop unrolling 8 per iteration and 8x8 blocks";
void pthread_rotate_blocking8(int dim, pixel *src, pixel *dst)
{

    int NUM_THREADS = 1;
    // only do multithreading if workload is high enough
    if (dim < 255)
    {
        rotate_blocking32_pointer(dim, src, dst);
        return;
    }
    else if (dim > 255 && dim < 511)
    {
        NUM_THREADS = 2;
    }
    else if (dim >= 500)
    {
        NUM_THREADS = 4;
    }

    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];

    int rows_per_thread = dim / NUM_THREADS;
    int remaining_rows = dim % NUM_THREADS;
    int current_row = 0;

    for (int i = 0; i < NUM_THREADS; i++)
    {
        thread_data[i].dim = dim;
        thread_data[i].src = src;
        thread_data[i].dst = dst;
        thread_data[i].start_row = current_row;
        thread_data[i].end_row = current_row + rows_per_thread + (i < remaining_rows ? 1 : 0);

        pthread_create(&threads[i], NULL, rotate_thread_blocking8, (void *)&thread_data[i]);

        current_row = thread_data[i].end_row;
    }

    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);
}

// Threading block 32x32
void *rotate_thread_blocking32(void *arg)
{
    ThreadData *data = (ThreadData *)arg;
    int dim = data->dim;
    pixel *src = data->src;
    pixel *dst = data->dst;

    for (int j = data->start_row; j < data->end_row; j += 32)
    {
        for (int i = 0; i < dim; i += 32)
        {
            for (int jj = 0; jj < 32; jj++)
            {
                int dstElement = RIDX(dim - 1 - (j + jj), i, dim);
                int srcElement = RIDX(i, j + jj, dim);

                dst[dstElement] = src[srcElement];
                dst[dstElement + 1] = src[srcElement + dim];
                dst[dstElement + 2] = src[srcElement + dim * 2];
                dst[dstElement + 3] = src[srcElement + dim * 3];
                dst[dstElement + 4] = src[srcElement + dim * 4];
                dst[dstElement + 5] = src[srcElement + dim * 5];
                dst[dstElement + 6] = src[srcElement + dim * 6];
                dst[dstElement + 7] = src[srcElement + dim * 7];
                dst[dstElement + 8] = src[srcElement + dim * 8];
                dst[dstElement + 9] = src[srcElement + dim * 9];
                dst[dstElement + 10] = src[srcElement + dim * 10];
                dst[dstElement + 11] = src[srcElement + dim * 11];
                dst[dstElement + 12] = src[srcElement + dim * 12];
                dst[dstElement + 13] = src[srcElement + dim * 13];
                dst[dstElement + 14] = src[srcElement + dim * 14];
                dst[dstElement + 15] = src[srcElement + dim * 15];
                dst[dstElement + 16] = src[srcElement + dim * 16];
                dst[dstElement + 17] = src[srcElement + dim * 17];
                dst[dstElement + 18] = src[srcElement + dim * 18];
                dst[dstElement + 19] = src[srcElement + dim * 19];
                dst[dstElement + 20] = src[srcElement + dim * 20];
                dst[dstElement + 21] = src[srcElement + dim * 21];
                dst[dstElement + 22] = src[srcElement + dim * 22];
                dst[dstElement + 23] = src[srcElement + dim * 23];
                dst[dstElement + 24] = src[srcElement + dim * 24];
                dst[dstElement + 25] = src[srcElement + dim * 25];
                dst[dstElement + 26] = src[srcElement + dim * 26];
                dst[dstElement + 27] = src[srcElement + dim * 27];
                dst[dstElement + 28] = src[srcElement + dim * 28];
                dst[dstElement + 29] = src[srcElement + dim * 29];
                dst[dstElement + 30] = src[srcElement + dim * 30];
                dst[dstElement + 31] = src[srcElement + dim * 31];
            }
        }
    }

    pthread_exit(NULL);
}

char pthread_rotate_blocking32_descr[] = "Pthread implementation with loop unrolling 32 per iteration and 32x32 blocks";
void pthread_rotate_blocking32(int dim, pixel *src, pixel *dst)
{
    int NUM_THREADS = 1;
    // only do multithreading if workload is high enough
    if (dim < 255)
    {
        rotate_blocking32(dim, src, dst);
        return;
    }
    else if (dim > 255 && dim < 511)
    {
        NUM_THREADS = 2;
    }
    else if (dim >= 500)
    {
        NUM_THREADS = 4;
    }

    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];

    int rows_per_thread = dim / NUM_THREADS;
    int remaining_rows = dim % NUM_THREADS;
    int current_row = 0;

    for (int i = 0; i < NUM_THREADS; i++)
    {
        thread_data[i].dim = dim;
        thread_data[i].src = src;
        thread_data[i].dst = dst;
        thread_data[i].start_row = current_row;
        thread_data[i].end_row = current_row + rows_per_thread + (i < remaining_rows ? 1 : 0);

        pthread_create(&threads[i], NULL, rotate_thread_blocking32, (void *)&thread_data[i]);

        current_row = thread_data[i].end_row;
    }

    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);
}

/*********************************************************************
 * register_rotate_t_functions - Register all of your different versions
 *     of the rotate_t kernel with the driver by calling the
 *     add_rotate_t_function() for each test function. When you run the
 *     driver program, it will test and report the performance of each
 *     registered test function.
 *********************************************************************/

void register_rotate_t_functions()
{
    add_rotate_t_function(&rotate_t, rotate_t_descr);
    add_rotate_t_function(&pthread_naive_rotate, pthread_naive_rotate_descr);
    add_rotate_t_function(&pthread_rotate_blocking8, pthread_rotate_blocking8_descr);
    add_rotate_t_function(&pthread_rotate_blocking32, pthread_rotate_blocking32_descr);
    /* ... Register additional test functions here */
}

/******************************************************************************
 * BLEND KERNEL
 *****************************************************************************/

// Your different versions of the blend kernel go here.

char naive_blend_descr[] = "blend_pixel: Naive baseline implementation";
void naive_blend(int dim, pixel *src, pixel *dst) // reads global variable `pixel &bgc`
{
    int i, j;

    for (i = 0; i < dim; i++)
        for (j = 0; j < dim; j++)
            blend_pixel(&src[RIDX(i, j, dim)], &dst[RIDX(i, j, dim)], &bgc); // `blend_pixel` defined in blend.c
}

char blend_descr[] = "blend: Current working version";
void blend(int dim, pixel *src, pixel *dst)
{
    naive_blend(dim, src, dst);
}

char blend_block8_unrolled8_descr[] = "Blocking 8x8, and unrolled loop with factor of 8";
void blend_block8_unrolled8(int dim, pixel *src, pixel *dst)
{
    int i, j, ii, jj;

    for (j = 0; j < dim; j += 8)
    {
        for (i = 0; i < dim; i += 8)
        {
            for (jj = j; jj < j + 8; jj++)
            {
                // Unroll the loop by a factor of 8
                for (ii = i; ii < i + 8; ii += 8)
                {
                    blend_pixel(&src[RIDX(ii, jj, dim)], &dst[RIDX(ii, jj, dim)], &bgc);
                    blend_pixel(&src[RIDX(ii + 1, jj, dim)], &dst[RIDX(ii + 1, jj, dim)], &bgc);
                    blend_pixel(&src[RIDX(ii + 2, jj, dim)], &dst[RIDX(ii + 2, jj, dim)], &bgc);
                    blend_pixel(&src[RIDX(ii + 3, jj, dim)], &dst[RIDX(ii + 3, jj, dim)], &bgc);
                    blend_pixel(&src[RIDX(ii + 4, jj, dim)], &dst[RIDX(ii + 4, jj, dim)], &bgc);
                    blend_pixel(&src[RIDX(ii + 5, jj, dim)], &dst[RIDX(ii + 5, jj, dim)], &bgc);
                    blend_pixel(&src[RIDX(ii + 6, jj, dim)], &dst[RIDX(ii + 6, jj, dim)], &bgc);
                    blend_pixel(&src[RIDX(ii + 7, jj, dim)], &dst[RIDX(ii + 7, jj, dim)], &bgc);
                }
            }
        }
    }
}

char blend_no_RIDX_descr[] = "Reuse of dim*dim, unroll by 4";
void blend_no_RIDX(int dim, pixel *src, pixel *dst)
{
    int i;
    // Move the calculation of `dim_squared` outside the loop for code motion
    int dim_squared = dim * dim;

    // Unroll the outer loop by a factor of 4 for better instruction pipelining
    for (i = 0; i < dim_squared; i += 4)
    {
        // Loop unrolling for the inner loop
        blend_pixel(&src[i], &dst[i], &bgc);
        blend_pixel(&src[i + 1], &dst[i + 1], &bgc);
        blend_pixel(&src[i + 2], &dst[i + 2], &bgc);
        blend_pixel(&src[i + 3], &dst[i + 3], &bgc);
    }
}

char blend_pointer_unroll4_descr[] = "with pointer unroll 4";
void blend_pointer_unroll4(int dim, pixel *src, pixel *dst)
{
    int i;

    // Move the calculation of `dim_squared` outside the loop for code motion
    int dim_squared = dim * dim;

    // Unroll the outer loop by a factor of 4 for better instruction pipelining
    for (i = 0; i < dim_squared; i += 4)
    {
        pixel *srcpointer = src + i;
        pixel *dstpointer = dst + i;
        // Loop unrolling for the inner loop
        blend_pixel(&srcpointer[0], &dstpointer[0], &bgc);
        blend_pixel(&srcpointer[1], &dstpointer[1], &bgc);
        blend_pixel(&srcpointer[2], &dstpointer[2], &bgc);
        blend_pixel(&srcpointer[3], &dstpointer[3], &bgc);
    }
}

char blend_unroll32_descr[] = "Blend - Unroll 32 - no pointers";
void blend_unroll32(int dim, pixel *src, pixel *dst)
{
    int i;

    // Move the calculation of `dim_squared` outside the loop for code motion
    int dim_squared = dim * dim;

    // Unroll the outer loop by a factor of 32
    for (i = 0; i < dim_squared; i += 32)
    {
        // Loop unrolling for the inner loop
        blend_pixel(&src[i], &dst[i], &bgc);
        blend_pixel(&src[i + 1], &dst[i + 1], &bgc);
        blend_pixel(&src[i + 2], &dst[i + 2], &bgc);
        blend_pixel(&src[i + 3], &dst[i + 3], &bgc);
        blend_pixel(&src[i + 4], &dst[i + 4], &bgc);
        blend_pixel(&src[i + 5], &dst[i + 5], &bgc);
        blend_pixel(&src[i + 6], &dst[i + 6], &bgc);
        blend_pixel(&src[i + 7], &dst[i + 7], &bgc);
        blend_pixel(&src[i + 8], &dst[i + 8], &bgc);
        blend_pixel(&src[i + 9], &dst[i + 9], &bgc);
        blend_pixel(&src[i + 10], &dst[i + 10], &bgc);
        blend_pixel(&src[i + 11], &dst[i + 11], &bgc);
        blend_pixel(&src[i + 12], &dst[i + 12], &bgc);
        blend_pixel(&src[i + 13], &dst[i + 13], &bgc);
        blend_pixel(&src[i + 14], &dst[i + 14], &bgc);
        blend_pixel(&src[i + 15], &dst[i + 15], &bgc);
        blend_pixel(&src[i + 16], &dst[i + 16], &bgc);
        blend_pixel(&src[i + 17], &dst[i + 17], &bgc);
        blend_pixel(&src[i + 18], &dst[i + 18], &bgc);
        blend_pixel(&src[i + 19], &dst[i + 19], &bgc);
        blend_pixel(&src[i + 20], &dst[i + 20], &bgc);
        blend_pixel(&src[i + 21], &dst[i + 21], &bgc);
        blend_pixel(&src[i + 22], &dst[i + 22], &bgc);
        blend_pixel(&src[i + 23], &dst[i + 23], &bgc);
        blend_pixel(&src[i + 24], &dst[i + 24], &bgc);
        blend_pixel(&src[i + 25], &dst[i + 25], &bgc);
        blend_pixel(&src[i + 26], &dst[i + 26], &bgc);
        blend_pixel(&src[i + 27], &dst[i + 27], &bgc);
        blend_pixel(&src[i + 28], &dst[i + 28], &bgc);
        blend_pixel(&src[i + 29], &dst[i + 29], &bgc);
        blend_pixel(&src[i + 30], &dst[i + 30], &bgc);
        blend_pixel(&src[i + 31], &dst[i + 31], &bgc);
    }
}

char blend_pointer_unrolled32_descr[] = "with pointer unrolled32";
void blend_pointer_unrolled32(int dim, pixel *src, pixel *dst)
{
    int i;
    int dim_squared = dim * dim;

    for (i = 0; i < dim_squared; i += 32)
    {
        pixel *srcPointer = src + i;
        pixel *dstPointer = dst + i;

        blend_pixel(&srcPointer[0], &dstPointer[0], &bgc);
        blend_pixel(&srcPointer[1], &dstPointer[1], &bgc);
        blend_pixel(&srcPointer[2], &dstPointer[2], &bgc);
        blend_pixel(&srcPointer[3], &dstPointer[3], &bgc);
        blend_pixel(&srcPointer[4], &dstPointer[4], &bgc);
        blend_pixel(&srcPointer[5], &dstPointer[5], &bgc);
        blend_pixel(&srcPointer[6], &dstPointer[6], &bgc);
        blend_pixel(&srcPointer[7], &dstPointer[7], &bgc);
        blend_pixel(&srcPointer[8], &dstPointer[8], &bgc);
        blend_pixel(&srcPointer[9], &dstPointer[9], &bgc);
        blend_pixel(&srcPointer[10], &dstPointer[10], &bgc);
        blend_pixel(&srcPointer[11], &dstPointer[11], &bgc);
        blend_pixel(&srcPointer[12], &dstPointer[12], &bgc);
        blend_pixel(&srcPointer[13], &dstPointer[13], &bgc);
        blend_pixel(&srcPointer[14], &dstPointer[14], &bgc);
        blend_pixel(&srcPointer[15], &dstPointer[15], &bgc);
        blend_pixel(&srcPointer[16], &dstPointer[16], &bgc);
        blend_pixel(&srcPointer[17], &dstPointer[17], &bgc);
        blend_pixel(&srcPointer[18], &dstPointer[18], &bgc);
        blend_pixel(&srcPointer[19], &dstPointer[19], &bgc);
        blend_pixel(&srcPointer[20], &dstPointer[20], &bgc);
        blend_pixel(&srcPointer[21], &dstPointer[21], &bgc);
        blend_pixel(&srcPointer[22], &dstPointer[22], &bgc);
        blend_pixel(&srcPointer[23], &dstPointer[23], &bgc);
        blend_pixel(&srcPointer[24], &dstPointer[24], &bgc);
        blend_pixel(&srcPointer[25], &dstPointer[25], &bgc);
        blend_pixel(&srcPointer[26], &dstPointer[26], &bgc);
        blend_pixel(&srcPointer[27], &dstPointer[27], &bgc);
        blend_pixel(&srcPointer[28], &dstPointer[28], &bgc);
        blend_pixel(&srcPointer[29], &dstPointer[29], &bgc);
        blend_pixel(&srcPointer[30], &dstPointer[30], &bgc);
        blend_pixel(&srcPointer[31], &dstPointer[31], &bgc);
    }
}

/*
 * register_blend_functions - Register all of your different versions
 *     of the blend kernel with the driver by calling the
 *     add_blend_function() for each test function.
 */
void register_blend_functions()
{
    add_blend_function(&blend, blend_descr);
    add_blend_function(&blend_block8_unrolled8, blend_block8_unrolled8_descr);
    add_blend_function(&blend_no_RIDX, blend_no_RIDX_descr);
    add_blend_function(&blend_pointer_unroll4, blend_pointer_unroll4_descr);
    add_blend_function(&blend_unroll32, blend_unroll32_descr);
    add_blend_function(&blend_pointer_unrolled32, blend_pointer_unrolled32_descr);
    /* ... Register additional test functions here */
}

/******************************************************************************
 * BLEND_V KERNEL
 *****************************************************************************/

// Your different versions of the blend_v kernel go here
// (i.e. with vectorization, aka. SIMD).

char blend_v_descr[] = "blend_v: Current working version BLEND V";
void blend_v(int dim, pixel *src, pixel *dst)
{
    naive_blend(dim, src, dst);
}

void print_avx2_vector(__m256 vector)
{
    float values[8];
    _mm256_storeu_ps(values, vector);

    for (int i = 0; i < 8; ++i)
    {
        printf("%f ", values[i]);
    }
    printf("\n");
}

void print_m256_pixel(__m256 var)
{
    float array[8];
    _mm256_storeu_ps(array, var);
    for (int i = 0; i < 8; i += 4)
    {
        printf("red: %f, green: %f, blue: %f, alpha: %f\n", array[i], array[i + 1], array[i + 2], array[i + 3]);
    }
}

void print_m256_elements(__m256 var)
{
    float array[8];
    _mm256_storeu_ps(array, var);
    for (int i = 0; i < 8; i++)
    {
        printf("Element %d: %f\n", i, array[i]);
    }
}

void blend_v_AVX2_2pix(int dim, pixel *src, pixel *dst)
{

    int printEnabled = 0;

    // bgc->red/blue/green
    __m256 bcgColors = _mm256_set_ps(
        bgc.alpha,
        bgc.blue,
        bgc.green,
        bgc.red,
        bgc.alpha,
        bgc.blue,
        bgc.green,
        bgc.red);

    if (printEnabled)
    {
        printf("real bgc: bgc red: %d. Bgc green: %d . Bgc blue %d \n", bgc.red, bgc.green, bgc.blue);
        printf("bcg vector:\n");
        print_m256_elements(bcgColors);
    }

    int size = dim * dim;

    float ushrtFast = 1.0f / USHRT_MAX; // calculate once, so i can reuse this with a multiply operation (strenght reduction, divide is more costly then multiply)

    // Process 2 pixels at once with AVX2
    for (int i = 0; i < size; i += 2)
    {

        if (printEnabled)
        {
            // To check actual answer:
            pixel realBefore[2];
            realBefore[0] = src[i];
            realBefore[1] = src[i + 1];
            pixel realAfter[2];
            realAfter[0] = dst[i];
            realAfter[1] = dst[i + 1];
            printf("Real answers:\n");
            printf("Before blend: Pixel[%d]: Red=%u, Green=%u, Blue=%u, Alpha=%u\n",
                   i, realBefore[0].red, realBefore[0].green, realBefore[0].blue, realBefore[0].alpha);
            printf("Before blend: Pixel[%d]: Red=%u, Green=%u, Blue=%u, Alpha=%u\n",
                   i + 1, realBefore[1].red, realBefore[1].green, realBefore[1].blue, realBefore[1].alpha);

            blend_pixel(&realBefore[0], &realAfter[0], &bgc);
            blend_pixel(&realBefore[1], &realAfter[1], &bgc);

            printf("after blend: Pixel[%d]: Red=%u, Green=%u, Blue=%u, Alpha=%u\n",
                   i, realAfter[0].red, realAfter[0].green, realAfter[0].blue, realAfter[0].alpha);
            printf("after blend: Pixel[%d]: Red=%u, Green=%u, Blue=%u, Alpha=%u\n",
                   i + 1, realAfter[1].red, realAfter[1].green, realAfter[1].blue, realAfter[1].alpha);
        }

        // Load 2 source pixels as floats
        __m256 src_pixels_256;
        src_pixels_256 = _mm256_set_ps(
            src[i].alpha, src[i].blue, src[i].green, src[i].red,
            src[i + 1].alpha, src[i + 1].blue, src[i + 1].green, src[i + 1].red);
        if (printEnabled)
        {
            printf("pixel vector: \n");
            print_m256_pixel(src_pixels_256);
        }

        // alpha
        float alphaone = (float)src[i].alpha * ushrtFast;
        float alphaTwo = (float)src[i + 1].alpha * ushrtFast;
        __m256 alpha_values = _mm256_set_ps(
            alphaone,
            alphaone,
            alphaone,
            alphaone,
            alphaTwo,
            alphaTwo,
            alphaTwo,
            alphaTwo);

        // alpha optimises
        if (printEnabled)
        {
            printf("alpha:\n");
            print_m256_elements(alpha_values);
        }

        //(1 - a)
        __m256 one_minus_alpha = _mm256_sub_ps(_mm256_set1_ps(1.0f), alpha_values);

        if (printEnabled)
        {
            printf("one minus alpha:\n");
            print_m256_elements(one_minus_alpha);
        }

        // Multiply 'alpha_values' with 'src_pixels_256'
        __m256 leftHandSide = _mm256_mul_ps(alpha_values, src_pixels_256);
        if (printEnabled)
        {
            printf("leftHandSide:\n");
            print_m256_elements(leftHandSide);
        }

        // Multiply '(1 - a)' with 'bgc' for each component
        __m256 rightHandSide = _mm256_mul_ps(one_minus_alpha, bcgColors);
        if (printEnabled)
        {
            printf("rightHandSide:\n");
            print_m256_elements(rightHandSide);
        }

        __m256 result = _mm256_add_ps(leftHandSide, rightHandSide);

        // todo, is there a faster way to store this back into dst??
        float result_array[8];
        _mm256_storeu_ps(result_array, result);

        // Store the result in the dst array
        dst[i].red = (unsigned short)result_array[4];
        dst[i].green = (unsigned short)result_array[5];
        dst[i].blue = (unsigned short)result_array[6];
        dst[i].alpha = USHRT_MAX; // Set alpha to USHRT_MAX (65535)

        dst[i + 1].red = (unsigned short)result_array[0];
        dst[i + 1].green = (unsigned short)result_array[1];
        dst[i + 1].blue = (unsigned short)result_array[2];
        dst[i + 1].alpha = USHRT_MAX; // Set alpha to USHRT_MAX (65535)

        if (printEnabled)
        {
            printf("Printing result: %d \n", i);
            printf("after vector blend: Pixel[%d]: Red=%u, Green=%u, Blue=%u, Alpha=%u\n",
                   i, dst[i].red, dst[i].green, dst[i].blue, dst[i].alpha);
            printf("Printing result: %d \n", i + 1);
            printf("after vector blend: Pixel[%d]: Red=%u, Green=%u, Blue=%u, Alpha=%u\n",
                   i + 1, dst[i + 1].red, dst[i + 1].green, dst[i + 1].blue, dst[i + 1].alpha);
        }
    }
}

void blend_v_AVX2_8pix(int dim, pixel *src, pixel *dst)
{

    int printEnabled = 0;
    int size = dim * dim;
    float ushrtFast = 1.0f / USHRT_MAX; // calculate once, so i can reuse this with a multiply operation (strenght reduction, divide is more costly then multiply)

    // Process 8 pixels at once with AVX2
    for (int i = 0; i < size; i += 8)
    {

        if (printEnabled)
        {
            // To check actual answer:
            pixel realBefore[8];
            realBefore[0] = src[i];
            realBefore[1] = src[i + 1];
            realBefore[2] = src[i + 2];
            realBefore[3] = src[i + 3];
            realBefore[4] = src[i + 4];
            realBefore[5] = src[i + 5];
            realBefore[6] = src[i + 6];
            realBefore[7] = src[i + 7];
            pixel realAfter[8];
            realAfter[0] = dst[i];
            realAfter[1] = dst[i + 1];
            realAfter[2] = dst[i + 2];
            realAfter[3] = dst[i + 3];
            realAfter[4] = dst[i + 4];
            realAfter[5] = dst[i + 5];
            realAfter[6] = dst[i + 6];
            realAfter[7] = dst[i + 7];
            printf("Real answers:\n");
            printf("Before blend: Pixel[%d]: Red=%u, Green=%u, Blue=%u, Alpha=%u\n",
                   i, realBefore[0].red, realBefore[0].green, realBefore[0].blue, realBefore[0].alpha);
            printf("Before blend: Pixel[%d]: Red=%u, Green=%u, Blue=%u, Alpha=%u\n",
                   i + 1, realBefore[1].red, realBefore[1].green, realBefore[1].blue, realBefore[1].alpha);
            printf("Before blend: Pixel[%d]: Red=%u, Green=%u, Blue=%u, Alpha=%u\n",
                   i + 2, realBefore[2].red, realBefore[2].green, realBefore[2].blue, realBefore[2].alpha);
            printf("Before blend: Pixel[%d]: Red=%u, Green=%u, Blue=%u, Alpha=%u\n",
                   i + 3, realBefore[3].red, realBefore[3].green, realBefore[3].blue, realBefore[3].alpha);
            printf("Before blend: Pixel[%d]: Red=%u, Green=%u, Blue=%u, Alpha=%u\n",
                   i + 4, realBefore[4].red, realBefore[4].green, realBefore[4].blue, realBefore[4].alpha);
            printf("Before blend: Pixel[%d]: Red=%u, Green=%u, Blue=%u, Alpha=%u\n",
                   i + 5, realBefore[5].red, realBefore[5].green, realBefore[5].blue, realBefore[5].alpha);
            printf("Before blend: Pixel[%d]: Red=%u, Green=%u, Blue=%u, Alpha=%u\n",
                   i + 6, realBefore[6].red, realBefore[6].green, realBefore[6].blue, realBefore[6].alpha);
            printf("Before blend: Pixel[%d]: Red=%u, Green=%u, Blue=%u, Alpha=%u\n",
                   i + 7, realBefore[7].red, realBefore[7].green, realBefore[7].blue, realBefore[7].alpha);

            blend_pixel(&realBefore[0], &realAfter[0], &bgc);
            blend_pixel(&realBefore[1], &realAfter[1], &bgc);
            blend_pixel(&realBefore[2], &realAfter[2], &bgc);
            blend_pixel(&realBefore[3], &realAfter[3], &bgc);
            blend_pixel(&realBefore[4], &realAfter[4], &bgc);
            blend_pixel(&realBefore[5], &realAfter[5], &bgc);
            blend_pixel(&realBefore[6], &realAfter[6], &bgc);
            blend_pixel(&realBefore[7], &realAfter[7], &bgc);

            printf("after blend: Pixel[%d]: Red=%u, Green=%u, Blue=%u, Alpha=%u\n",
                   i, realAfter[0].red, realAfter[0].green, realAfter[0].blue, realAfter[0].alpha);
            printf("after blend: Pixel[%d]: Red=%u, Green=%u, Blue=%u, Alpha=%u\n",
                   i + 1, realAfter[1].red, realAfter[1].green, realAfter[1].blue, realAfter[1].alpha);
            printf("after blend: Pixel[%d]: Red=%u, Green=%u, Blue=%u, Alpha=%u\n",
                   i + 2, realAfter[2].red, realAfter[2].green, realAfter[2].blue, realAfter[2].alpha);
            printf("after blend: Pixel[%d]: Red=%u, Green=%u, Blue=%u, Alpha=%u\n",
                   i + 3, realAfter[3].red, realAfter[3].green, realAfter[3].blue, realAfter[3].alpha);
            printf("after blend: Pixel[%d]: Red=%u, Green=%u, Blue=%u, Alpha=%u\n",
                   i + 4, realAfter[4].red, realAfter[4].green, realAfter[4].blue, realAfter[4].alpha);
            printf("after blend: Pixel[%d]: Red=%u, Green=%u, Blue=%u, Alpha=%u\n",
                   i + 5, realAfter[5].red, realAfter[5].green, realAfter[5].blue, realAfter[5].alpha);
            printf("after blend: Pixel[%d]: Red=%u, Green=%u, Blue=%u, Alpha=%u\n",
                   i + 6, realAfter[6].red, realAfter[6].green, realAfter[6].blue, realAfter[6].alpha);
            printf("after blend: Pixel[%d]: Red=%u, Green=%u, Blue=%u, Alpha=%u\n",
                   i + 7, realAfter[7].red, realAfter[7].green, realAfter[7].blue, realAfter[7].alpha);
        }

        // load 8 src red
        __m256 src_red = _mm256_set_ps(
            src[i].red, src[i + 1].red, src[i + 2].red, src[i + 3].red, src[i + 4].red, src[i + 5].red, src[i + 6].red, src[i + 7].red);

        // Load 8 src green
        __m256 src_green = _mm256_set_ps(
            src[i].green, src[i + 1].green, src[i + 2].green, src[i + 3].green, src[i + 4].green, src[i + 5].green, src[i + 6].green, src[i + 7].green);

        // load 8 src blue
        __m256 src_blue = _mm256_set_ps(
            src[i].blue, src[i + 1].blue, src[i + 2].blue, src[i + 3].blue, src[i + 4].blue, src[i + 5].blue, src[i + 6].blue, src[i + 7].blue);

        // load 8 src alpha
        __m256 src_alpha = _mm256_set_ps(
            src[i].alpha, src[i + 1].alpha, src[i + 2].alpha, src[i + 3].alpha, src[i + 4].alpha, src[i + 5].alpha, src[i + 6].alpha, src[i + 7].alpha);

        if (printEnabled)
        {
            printf("pixel vectors: \n");
            print_m256_pixel(src_red);
        }

        // alpha / USHRT MAX , but using multiply (strenght reduction, multiply cheaper than divide)
        __m256 alpha_divided_ushrt = _mm256_mul_ps(_mm256_set1_ps(ushrtFast), src_alpha);

        if (printEnabled)
        {
            printf("alpha:\n");
            print_m256_elements(alpha_divided_ushrt);
        }

        // OneMinusAlpha(1 - a)
        __m256 one_minus_alpha = _mm256_sub_ps(_mm256_set1_ps(1.0f), alpha_divided_ushrt);

        if (printEnabled)
        {
            printf("one minus alpha:\n");
            print_m256_elements(one_minus_alpha);
        }

        // leftHandSide Red
        __m256 leftHandSideRed = _mm256_mul_ps(alpha_divided_ushrt, src_red);

        // leftHandSide Green
        __m256 leftHandSideGreen = _mm256_mul_ps(alpha_divided_ushrt, src_green);

        // leftHandSide blue
        __m256 leftHandSideBlue = _mm256_mul_ps(alpha_divided_ushrt, src_blue);

        if (printEnabled)
        {
            printf("leftHandSideRed:\n");
            print_m256_elements(leftHandSideRed);
            printf("leftHandSideGreen:\n");
            print_m256_elements(leftHandSideGreen);
            printf("leftHandSideBlue:\n");
            print_m256_elements(leftHandSideBlue);
        }

        // Right hand side: (one_minus_alpha) multiply with relevant bgc
        //'(1 - a)' *'bgc.red
        __m256 rightHandSideRed = _mm256_mul_ps(one_minus_alpha, _mm256_set1_ps((float)bgc.red));

        //'(1 - a)' *'bgc.green
        __m256 rightHandSideGreen = _mm256_mul_ps(one_minus_alpha, _mm256_set1_ps((float)bgc.green));

        //'(1 - a)' *'bgc.blue
        __m256 rightHandSideBlue = _mm256_mul_ps(one_minus_alpha, _mm256_set1_ps((float)bgc.blue));

        if (printEnabled)
        {
            printf("rightHandSideRed:\n");
            print_m256_elements(rightHandSideRed);
            printf("rightHandSideGreen:\n");
            print_m256_elements(rightHandSideGreen);
            printf("rightHandSideBlue:\n");
            print_m256_elements(rightHandSideBlue);
        }

        // Result red
        __m256 resultRed = _mm256_add_ps(leftHandSideRed, rightHandSideRed);

        // result green
        __m256 resultGreen = _mm256_add_ps(leftHandSideGreen, rightHandSideGreen);

        // result blue
        __m256 resultBlue = _mm256_add_ps(leftHandSideBlue, rightHandSideBlue);

        // todo, is there a faster way to store this back into dst??
        float red_result_array[8];
        _mm256_storeu_ps(red_result_array, resultRed);

        float green_result_array[8];
        _mm256_storeu_ps(green_result_array, resultGreen);

        float blue_result_array[8];
        _mm256_storeu_ps(blue_result_array, resultBlue);

        // Store the result in the dst array
        // Red
        dst[i].red = (unsigned short)red_result_array[7];
        dst[i + 1].red = (unsigned short)red_result_array[6];
        dst[i + 2].red = (unsigned short)red_result_array[5];
        dst[i + 3].red = (unsigned short)red_result_array[4];
        dst[i + 4].red = (unsigned short)red_result_array[3];
        dst[i + 5].red = (unsigned short)red_result_array[2];
        dst[i + 6].red = (unsigned short)red_result_array[1];
        dst[i + 7].red = (unsigned short)red_result_array[0];

        // Green
        dst[i].green = (unsigned short)green_result_array[7];
        dst[i + 1].green = (unsigned short)green_result_array[6];
        dst[i + 2].green = (unsigned short)green_result_array[5];
        dst[i + 3].green = (unsigned short)green_result_array[4];
        dst[i + 4].green = (unsigned short)green_result_array[3];
        dst[i + 5].green = (unsigned short)green_result_array[2];
        dst[i + 6].green = (unsigned short)green_result_array[1];
        dst[i + 7].green = (unsigned short)green_result_array[0];

        // blue
        dst[i].blue = (unsigned short)blue_result_array[7];
        dst[i + 1].blue = (unsigned short)blue_result_array[6];
        dst[i + 2].blue = (unsigned short)blue_result_array[5];
        dst[i + 3].blue = (unsigned short)blue_result_array[4];
        dst[i + 4].blue = (unsigned short)blue_result_array[3];
        dst[i + 5].blue = (unsigned short)blue_result_array[2];
        dst[i + 6].blue = (unsigned short)blue_result_array[1];
        dst[i + 7].blue = (unsigned short)blue_result_array[0];

        // alpha - Set alpha to USHRT_MAX (65535)
        dst[i].alpha = USHRT_MAX;
        dst[i + 1].alpha = USHRT_MAX;
        dst[i + 2].alpha = USHRT_MAX;
        dst[i + 3].alpha = USHRT_MAX;
        dst[i + 4].alpha = USHRT_MAX;
        dst[i + 5].alpha = USHRT_MAX;
        dst[i + 6].alpha = USHRT_MAX;
        dst[i + 7].alpha = USHRT_MAX;

        if (printEnabled)
        {
            printf("Printing result: %d \n", i);
            printf("after vector blend: Pixel[%d]: Red=%u, Green=%u, Blue=%u, Alpha=%u\n",
                   i, dst[i].red, dst[i].green, dst[i].blue, dst[i].alpha);
            printf("Printing result: %d \n", i + 1);
            printf("after vector blend: Pixel[%d]: Red=%u, Green=%u, Blue=%u, Alpha=%u\n",
                   i + 1, dst[i + 1].red, dst[i + 1].green, dst[i + 1].blue, dst[i + 1].alpha);
        }
    }
}

void blend_v_AVX2_8pix_noprint(int dim, pixel *src, pixel *dst)
{
    int size = dim * dim;
    float ushrtFast = 1.0f / USHRT_MAX; // calculate once, so i can reuse this with a multiply operation (strenght reduction, divide is more costly then multiply)

    // Process 8 pixels at once with AVX2
    for (int i = 0; i < size; i += 8)
    {
        // load 8 src red
        __m256 src_red = _mm256_set_ps(
            src[i + 7].red, src[i + 6].red, src[i + 5].red, src[i + 4].red, src[i + 3].red, src[i + 2].red, src[i + 1].red, src[i].red);

        // Load 8 src green
        __m256 src_green = _mm256_set_ps(
            src[i + 7].green, src[i + 6].green, src[i + 5].green, src[i + 4].green, src[i + 3].green, src[i + 2].green, src[i + 1].green, src[i].green);

        // load 8 src blue
        __m256 src_blue = _mm256_set_ps(
            src[i + 7].blue, src[i + 6].blue, src[i + 5].blue, src[i + 4].blue, src[i + 3].blue, src[i + 2].blue, src[i + 1].blue, src[i].blue);

        // load 8 src alpha
        __m256 src_alpha = _mm256_set_ps(
            src[i + 7].alpha, src[i + 6].alpha, src[i + 5].alpha, src[i + 4].alpha, src[i + 3].alpha, src[i + 2].alpha, src[i + 1].alpha, src[i].alpha);

        // alpha / USHRT MAX , but using multiply (strenght reduction, multiply cheaper than divide)
        __m256 alpha_divided_ushrt = _mm256_mul_ps(_mm256_set1_ps(ushrtFast), src_alpha);

        // OneMinusAlpha(1 - a)
        __m256 one_minus_alpha = _mm256_sub_ps(_mm256_set1_ps(1.0f), alpha_divided_ushrt);

        // leftHandSide Red
        __m256 leftHandSideRed = _mm256_mul_ps(alpha_divided_ushrt, src_red);

        // leftHandSide Green
        __m256 leftHandSideGreen = _mm256_mul_ps(alpha_divided_ushrt, src_green);

        // leftHandSide blue
        __m256 leftHandSideBlue = _mm256_mul_ps(alpha_divided_ushrt, src_blue);

        // Right hand side: (one_minus_alpha) multiply with relevant bgc
        //'(1 - a)' * bgc.red
        __m256 rightHandSideRed = _mm256_mul_ps(one_minus_alpha, _mm256_set1_ps((float)bgc.red));

        //'(1 - a)' * bgc.green
        __m256 rightHandSideGreen = _mm256_mul_ps(one_minus_alpha, _mm256_set1_ps((float)bgc.green));

        //'(1 - a)' * bgc.blue
        __m256 rightHandSideBlue = _mm256_mul_ps(one_minus_alpha, _mm256_set1_ps((float)bgc.blue));

        // Result red
        __m256 resultRed = _mm256_add_ps(leftHandSideRed, rightHandSideRed);

        // result green
        __m256 resultGreen = _mm256_add_ps(leftHandSideGreen, rightHandSideGreen);

        // result blue
        __m256 resultBlue = _mm256_add_ps(leftHandSideBlue, rightHandSideBlue);

        // todo, is there a faster way to store this back into dst??
        float red_result_array[8];
        _mm256_storeu_ps(red_result_array, resultRed);

        float green_result_array[8];
        _mm256_storeu_ps(green_result_array, resultGreen);

        float blue_result_array[8];
        _mm256_storeu_ps(blue_result_array, resultBlue);

        // Store the result in the dst array
        // Red
        dst[i].red = (unsigned short)red_result_array[0];
        dst[i + 1].red = (unsigned short)red_result_array[1];
        dst[i + 2].red = (unsigned short)red_result_array[2];
        dst[i + 3].red = (unsigned short)red_result_array[3];
        dst[i + 4].red = (unsigned short)red_result_array[4];
        dst[i + 5].red = (unsigned short)red_result_array[5];
        dst[i + 6].red = (unsigned short)red_result_array[6];
        dst[i + 7].red = (unsigned short)red_result_array[7];

        // Green
        dst[i].green = (unsigned short)green_result_array[0];
        dst[i + 1].green = (unsigned short)green_result_array[1];
        dst[i + 2].green = (unsigned short)green_result_array[2];
        dst[i + 3].green = (unsigned short)green_result_array[3];
        dst[i + 4].green = (unsigned short)green_result_array[4];
        dst[i + 5].green = (unsigned short)green_result_array[5];
        dst[i + 6].green = (unsigned short)green_result_array[6];
        dst[i + 7].green = (unsigned short)green_result_array[7];

        // blue
        dst[i].blue = (unsigned short)blue_result_array[0];
        dst[i + 1].blue = (unsigned short)blue_result_array[1];
        dst[i + 2].blue = (unsigned short)blue_result_array[2];
        dst[i + 3].blue = (unsigned short)blue_result_array[3];
        dst[i + 4].blue = (unsigned short)blue_result_array[4];
        dst[i + 5].blue = (unsigned short)blue_result_array[5];
        dst[i + 6].blue = (unsigned short)blue_result_array[6];
        dst[i + 7].blue = (unsigned short)blue_result_array[7];

        // alpha - Set alpha to USHRT_MAX (65535)
        dst[i].alpha = USHRT_MAX;
        dst[i + 1].alpha = USHRT_MAX;
        dst[i + 2].alpha = USHRT_MAX;
        dst[i + 3].alpha = USHRT_MAX;
        dst[i + 4].alpha = USHRT_MAX;
        dst[i + 5].alpha = USHRT_MAX;
        dst[i + 6].alpha = USHRT_MAX;
        dst[i + 7].alpha = USHRT_MAX;
    }
}

/*
 * register_blend_v_functions - Register all of your different versions
 *     of the blend_v kernel with the driver by calling the
 *     add_blend_function() for each test function.
 */
void register_blend_v_functions()
{
    add_blend_v_function(&blend_v, blend_v_descr);
    add_blend_v_function(&blend_v_AVX2_2pix, "Blend vectorization, with 2 pixel at a time");
    add_blend_v_function(&blend_v_AVX2_8pix, "Blend vectorization, with 8 pixel at a time");
    add_blend_v_function(&blend_v_AVX2_8pix_noprint, "Blend vectorization, with 8 pixel at a time, no prints");
    /* ... Register additional test functions here */
}

/******************************************************************************
 * SMOOTH KERNEL
 *****************************************************************************/

// Your different versions of the smooth kernel go here

/*
 * naive_smooth - The naive baseline version of smooth
 */
char naive_smooth_descr[] = "naive_smooth: Naive baseline implementation";
void naive_smooth(int dim, pixel *src, pixel *dst)
{
    int i, j;

    for (i = 0; i < dim; i++)
        for (j = 0; j < dim; j++)
            dst[RIDX(i, j, dim)] = avg(dim, i, j, src); // `avg` defined in smooth.c
}

char smooth_block_8_descr[] = "smooth with 8*8 block";
void smooth_block_8(int dim, pixel *src, pixel *dst)
{
    int i, j, ii, jj;

    for (j = 0; j < dim; j += 8)
    {
        for (i = 0; i < dim; i += 8)
        {
            // Unroll the loop by a factor of 8
            for (jj = j; jj < j + 8; jj++)
            {
                for (ii = i; ii < i + 8; ii += 8)
                {
                    dst[RIDX(ii, jj, dim)] = avg(dim, ii, jj, src);
                    dst[RIDX(ii + 1, jj, dim)] = avg(dim, ii + 1, jj, src);
                    dst[RIDX(ii + 2, jj, dim)] = avg(dim, ii + 2, jj, src);
                    dst[RIDX(ii + 3, jj, dim)] = avg(dim, ii + 3, jj, src);
                    dst[RIDX(ii + 4, jj, dim)] = avg(dim, ii + 4, jj, src);
                    dst[RIDX(ii + 5, jj, dim)] = avg(dim, ii + 5, jj, src);
                    dst[RIDX(ii + 6, jj, dim)] = avg(dim, ii + 6, jj, src);
                    dst[RIDX(ii + 7, jj, dim)] = avg(dim, ii + 7, jj, src);
                }
            }
        }
    }
}

// Created myself, but some inspiration from https://github.com/Zhenye-Na/CSAPP-Labs/blob/master/labs/Lab7-Performance%20Lab/perflab/kernels.c
char smooth_faster_descr[] = "Reusing variables, shifting and unrolled, special Wall/corner cases";
void smooth_faster(int dim, pixel *src, pixel *dst)
{
    // +-------------+--------+--------+--------------+
    // |  Left Top   |Wall Top|Wall Top|  Right Top   |
    // +-------------+--------+--------+--------------+
    // |   Wall Left |   mid  |   mid  |  Wall Right  |
    // +-------------+--------+--------+--------------+
    // |   Wall Left |   mid  |   mid  |  Wall Right  |
    // +-------------+--------+--------+--------------+
    // | Left Bottom |Wall Bot|Wall Bot| Right Bottom |
    // +-------------+--------+--------+--------------+

    // reusable variables:
    int curr;
    const int dim_minus_1 = dim - 1;

    // Left Top
    const int dim_plus_1 = dim + 1;
    dst[0].red = (src[0].red + src[1].red + src[dim].red + src[dim_plus_1].red) >> 2;
    dst[0].green = (src[0].green + src[1].green + src[dim].green + src[dim_plus_1].green) >> 2;
    dst[0].blue = (src[0].blue + src[1].blue + src[dim].blue + src[dim_plus_1].blue) >> 2;
    dst[0].alpha = (src[0].alpha + src[1].alpha + src[dim].alpha + src[dim_plus_1].alpha) >> 2;

    // Right Top
    curr = dim_minus_1;
    const int rightTopCurrMinusOne = curr - 1;
    const int rightTopCurrPlusDim = curr + dim;
    const int rightTopCurrPlusDimMinusOne = curr + dim - 1;
    dst[curr].red = (src[curr].red + src[rightTopCurrMinusOne].red + src[rightTopCurrPlusDimMinusOne].red + src[rightTopCurrPlusDim].red) >> 2;
    dst[curr].green = (src[curr].green + src[rightTopCurrMinusOne].green + src[rightTopCurrPlusDimMinusOne].green + src[rightTopCurrPlusDim].green) >> 2;
    dst[curr].blue = (src[curr].blue + src[rightTopCurrMinusOne].blue + src[rightTopCurrPlusDimMinusOne].blue + src[rightTopCurrPlusDim].blue) >> 2;
    dst[curr].alpha = (src[curr].alpha + src[rightTopCurrMinusOne].alpha + src[rightTopCurrPlusDimMinusOne].alpha + src[rightTopCurrPlusDim].alpha) >> 2;

    // Left Bottom
    curr *= dim;
    const int leftCurrPlusOne = curr + 1;
    const int leftCurrMinusDim = curr - dim;
    const int leftCurrMinusDimPlusOne = curr - dim + 1;
    dst[curr].red = (src[curr].red + src[leftCurrPlusOne].red + src[leftCurrMinusDim].red + src[leftCurrMinusDimPlusOne].red) >> 2;
    dst[curr].green = (src[curr].green + src[leftCurrPlusOne].green + src[leftCurrMinusDim].green + src[leftCurrMinusDimPlusOne].green) >> 2;
    dst[curr].blue = (src[curr].blue + src[leftCurrPlusOne].blue + src[leftCurrMinusDim].blue + src[leftCurrMinusDimPlusOne].blue) >> 2;
    dst[curr].alpha = (src[curr].alpha + src[leftCurrPlusOne].alpha + src[leftCurrMinusDim].alpha + src[leftCurrMinusDimPlusOne].alpha) >> 2;

    // Right Bottom
    curr += dim_minus_1;
    const int rightBottomCurrMinusOne = curr - 1;
    const int rightBottomCurrMinusDim = curr - dim;
    const int rightBottomCurrMinusDimMinusOne = curr - dim - 1;
    dst[curr].red = (src[curr].red + src[rightBottomCurrMinusOne].red + src[rightBottomCurrMinusDim].red + src[rightBottomCurrMinusDimMinusOne].red) >> 2;
    dst[curr].green = (src[curr].green + src[rightBottomCurrMinusOne].green + src[rightBottomCurrMinusDim].green + src[rightBottomCurrMinusDimMinusOne].green) >> 2;
    dst[curr].blue = (src[curr].blue + src[rightBottomCurrMinusOne].blue + src[rightBottomCurrMinusDim].blue + src[rightBottomCurrMinusDimMinusOne].blue) >> 2;
    dst[curr].alpha = (src[curr].alpha + src[rightBottomCurrMinusOne].alpha + src[rightBottomCurrMinusDim].alpha + src[rightBottomCurrMinusDimMinusOne].alpha) >> 2;

    int ii, jj, limit;
    // Wall Top
    limit = dim_minus_1;
    for (ii = 1; ii < limit; ii++)
    {
        const int iiMinusOne = ii - 1;
        const int iiPlusOne = ii + 1;
        const int iiPlusDim = ii + dim;
        const int iiPlusDimMinusOne = ii + dim - 1;
        const int iiPlusDimPlusOne = ii + dim + 1;
        dst[ii].red = (src[ii].red + src[iiMinusOne].red + src[iiPlusOne].red + src[iiPlusDim].red + src[iiPlusDimMinusOne].red + src[iiPlusDimPlusOne].red) / 6;
        dst[ii].green = (src[ii].green + src[iiMinusOne].green + src[iiPlusOne].green + src[iiPlusDim].green + src[iiPlusDimMinusOne].green + src[iiPlusDimPlusOne].green) / 6;
        dst[ii].blue = (src[ii].blue + src[iiMinusOne].blue + src[iiPlusOne].blue + src[iiPlusDim].blue + src[iiPlusDimMinusOne].blue + src[iiPlusDimPlusOne].blue) / 6;
        dst[ii].alpha = (src[ii].alpha + src[iiMinusOne].alpha + src[iiPlusOne].alpha + src[iiPlusDim].alpha + src[iiPlusDimMinusOne].alpha + src[iiPlusDimPlusOne].alpha) / 6;
    }

    // Wall Bot
    limit = dim * dim - 1;
    for (ii = (dim - 1) * dim + 1; ii < limit; ii++)
    {
        const int iiMinusOne = ii - 1;
        const int iiPlusOne = ii + 1;
        const int iiMinusDim = ii - dim;
        const int iiMinusDimMinusOne = ii - dim - 1;
        const int iiMinusDimPlusOne = ii - dim + 1;
        dst[ii].red = (src[ii].red + src[iiMinusOne].red + src[iiPlusOne].red + src[iiMinusDim].red + src[iiMinusDimMinusOne].red + src[iiMinusDimPlusOne].red) / 6;
        dst[ii].green = (src[ii].green + src[iiMinusOne].green + src[iiPlusOne].green + src[iiMinusDim].green + src[iiMinusDimMinusOne].green + src[iiMinusDimPlusOne].green) / 6;
        dst[ii].blue = (src[ii].blue + src[iiMinusOne].blue + src[iiPlusOne].blue + src[iiMinusDim].blue + src[iiMinusDimMinusOne].blue + src[iiMinusDimPlusOne].blue) / 6;
        dst[ii].alpha = (src[ii].alpha + src[iiMinusOne].alpha + src[iiPlusOne].alpha + src[iiMinusDim].alpha + src[iiMinusDimMinusOne].alpha + src[iiMinusDimPlusOne].alpha) / 6;
    }

    // Wall Left
    limit = dim * (dim - 1);
    for (jj = dim; jj < limit; jj += dim)
    {
        const int jjPlusOne = jj + 1;
        const int jjPlusDim = jj + dim;
        const int jjPlusDimPlusOne = jj + dim + 1;
        const int jjminusDim = jj - dim;
        const int jjminusDimPlusOne = jj - dim + 1;
        dst[jj].red = (src[jj].red + src[jjPlusOne].red + src[jjminusDim].red + src[jjminusDimPlusOne].red + src[jjPlusDim].red + src[jjPlusDimPlusOne].red) / 6;
        dst[jj].green = (src[jj].green + src[jjPlusOne].green + src[jjminusDim].green + src[jjminusDimPlusOne].green + src[jjPlusDim].green + src[jjPlusDimPlusOne].green) / 6;
        dst[jj].blue = (src[jj].blue + src[jjPlusOne].blue + src[jjminusDim].blue + src[jjminusDimPlusOne].blue + src[jjPlusDim].blue + src[jjPlusDimPlusOne].blue) / 6;
        dst[jj].alpha = (src[jj].alpha + src[jjPlusOne].alpha + src[jjminusDim].alpha + src[jjminusDimPlusOne].alpha + src[jjPlusDim].alpha + src[jjPlusDimPlusOne].alpha) / 6;
    }
    // Wall Right
    for (jj = 2 * dim - 1; jj < limit; jj += dim)
    {
        const int jjMinusOne = jj - 1;
        const int jjMinusDim = jj - dim;
        const int jjMinusDimMinusOne = jj - dim - 1;
        const int jjPlusDim = jj + dim;
        const int jjPlusDimMinusOne = jj + dim - 1;
        dst[jj].red = (src[jj].red + src[jjMinusOne].red + src[jjMinusDim].red + src[jjMinusDimMinusOne].red + src[jjPlusDim].red + src[jjPlusDimMinusOne].red) / 6;
        dst[jj].green = (src[jj].green + src[jjMinusOne].green + src[jjMinusDim].green + src[jjMinusDimMinusOne].green + src[jjPlusDim].green + src[jjPlusDimMinusOne].green) / 6;
        dst[jj].blue = (src[jj].blue + src[jjMinusOne].blue + src[jjMinusDim].blue + src[jjMinusDimMinusOne].blue + src[jjPlusDim].blue + src[jjPlusDimMinusOne].blue) / 6;
        dst[jj].alpha = (src[jj].alpha + src[jjMinusOne].alpha + src[jjMinusDim].alpha + src[jjMinusDimMinusOne].alpha + src[jjPlusDim].alpha + src[jjPlusDimMinusOne].alpha) / 6;
    }

    // Middle Pixels
    int i, j;
    int currMinusOne, currPlusOne, currMinusDim, currPlusDim, currMinusDimMinusOne, currMinusDimPlusOne, currPlusDimMinusOne, currPlusDimPlusOne;
    for (i = 1; i < dim_minus_1; i++)
    {
        for (j = 1; j < dim_minus_1; j++)
        {
            curr = i * dim + j;
            currMinusOne = curr - 1;
            currPlusOne = curr + 1;
            currMinusDim = curr - dim;
            currPlusDim = curr + dim;
            currMinusDimMinusOne = curr - dim - 1;
            currMinusDimPlusOne = curr - dim + 1;
            currPlusDimMinusOne = curr + dim - 1;
            currPlusDimPlusOne = curr + dim + 1;

            dst[curr].red = (src[curr].red + src[currMinusOne].red + src[currPlusOne].red + src[currMinusDim].red + src[currMinusDimMinusOne].red + src[currMinusDimPlusOne].red + src[currPlusDim].red + src[currPlusDimMinusOne].red + src[currPlusDimPlusOne].red) / 9;

            dst[curr].green = (src[curr].green + src[currMinusOne].green + src[currPlusOne].green + src[currMinusDim].green + src[currMinusDimMinusOne].green + src[currMinusDimPlusOne].green + src[currPlusDim].green + src[currPlusDimMinusOne].green + src[currPlusDimPlusOne].green) / 9;

            dst[curr].blue = (src[curr].blue + src[currMinusOne].blue + src[currPlusOne].blue + src[currMinusDim].blue + src[currMinusDimMinusOne].blue + src[currMinusDimPlusOne].blue + src[currPlusDim].blue + src[currPlusDimMinusOne].blue + src[currPlusDimPlusOne].blue) / 9;

            dst[curr].alpha = (src[curr].alpha + src[currMinusOne].alpha + src[currPlusOne].alpha + src[currMinusDim].alpha + src[currMinusDimMinusOne].alpha + src[currMinusDimPlusOne].alpha + src[currPlusDim].alpha + src[currPlusDimMinusOne].alpha + src[currPlusDimPlusOne].alpha) / 9;
        }
    }
}

char smooth_descr[] = "smooth: Current working version SMOOTH";
void smooth(int dim, pixel *src, pixel *dst)
{
    naive_smooth(dim, src, dst);
}

/*
 * register_smooth_functions - Register all of your different versions
 *     of the smooth kernel with the driver by calling the
 *     add_smooth_function() for each test function.
 */

void register_smooth_functions()
{
    add_smooth_function(&naive_smooth, naive_smooth_descr);
    add_smooth_function(&smooth_block_8, smooth_block_8_descr);
    add_smooth_function(&smooth_faster, smooth_faster_descr);
    /* ... Register additional test functions here */
}
