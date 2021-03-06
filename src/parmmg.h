/* =============================================================================
**  This file is part of the parmmg software package for parallel tetrahedral
**  mesh modification.
**  Copyright (c) Bx INP/Inria/UBordeaux, 2017-
**
**  parmmg is free software: you can redistribute it and/or modify it
**  under the terms of the GNU Lesser General Public License as published
**  by the Free Software Foundation, either version 3 of the License, or
**  (at your option) any later version.
**
**  parmmg is distributed in the hope that it will be useful, but WITHOUT
**  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
**  FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
**  License for more details.
**
**  You should have received a copy of the GNU Lesser General Public
**  License and of the GNU General Public License along with parmmg (in
**  files COPYING.LESSER and COPYING). If not, see
**  <http://www.gnu.org/licenses/>. Please read their terms carefully and
**  use this copy of the parmmg distribution only if you accept them.
** =============================================================================
*/

/**
 * \file parmmg.h
 * \brief internal functions headers for parmmg
 * \author Cécile Dobrzynski (Bx INP/Inria)
 * \author Algiane Froehly (Inria)
 * \version 5
 * \copyright GNU Lesser General Public License.
 */

#ifndef _PARMMG_H
#define _PARMMG_H

#include <stdint.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <string.h>
#include <signal.h>
#include <ctype.h>
#include <float.h>
#include <math.h>
#include <mpi_pmmg.h>

#include "libparmmg.h"
#include "interpmesh_pmmg.h"
#include "mmg3d.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \def PMMG_NUL
 *
 * Null value
 *
 */
#define PMMG_NUL     0

/**
 * \def PMMG_NITER
 *
 * Default number of iterations
 *
 */
#define PMMG_NITER   3

/**
 * \def PMMG_IMPRIM
 *
 * Default verbosity
 *
 */
#define PMMG_IMPRIM   1

/**
 * \def PMMG_MMG_IMPRIM
 *
 * Default verbosity for Mmg
 *
 */
#define PMMG_MMG_IMPRIM   -1

/**
 *
 * Size of quality histogram arrays
 *
 */
#define PMMG_QUAL_HISSIZE 5

/**
 *
 * Size of mpi datatype for quality histo computation
 *
 */
#define PMMG_QUAL_MPISIZE 4

/**
 *
 * Size of mpi datatype for length histo computation
 *
 */
#define PMMG_LENSTATS_MPISIZE 12

/**
 *
 * print input quality histogram
 *
 */
#define PMMG_INQUA 1

/**
 *
 * print output quality histogram
 *
 */
#define PMMG_OUTQUA 2

/**
 *
 * no verbosity for pmmg library
 *
 */
#define PMMG_VERB_NO -1

/**
 *
 * minimal verbosity for pmmg library: print library version and duration
 *
 */
#define PMMG_VERB_VERSION 0

/**
 *
 * low verbosity for pmmg library: print main steps + quality histo overview
 *
 */
#define PMMG_VERB_QUAL 1

/**
 *
 * average verbosity for pmmg library: add parmmg steps information
 *
 */
#define PMMG_VERB_STEPS 3

/**
 *
 * average verbosity for pmmg library: add waves information
 *
 */
#define PMMG_VERB_ITWAVES 4

/**
 *
 * detailed verbosity for pmmg library: add detailed quality histo
 *
 */
#define PMMG_VERB_DETQUAL 5

/**
 *
 * Split groups for redistribution (split_grps)
 *
 */
#define PMMG_GRPSPL_DISTR_TARGET 1

/**
 *
 * Split groups for mmg (split_grps)
 *
 */
#define PMMG_GRPSPL_MMG_TARGET 2

/**
 *
 * Use custom partitioning saved in the reference field (1=yes, 0=no)
 *
 */
#define PMMG_PREDEF_PART 0

/**
 * \enum PMMG_Format
 * \brief Type of supported file format
 */
enum PMMG_Format {
  PMMG_FMT_MeditASCII  = MMG5_FMT_MeditASCII, /*!< ASCII Medit (.mesh) */
  PMMG_FMT_MeditBinary = MMG5_FMT_MeditBinary,/*!< Binary Medit (.meshb) */
  PMMG_FMT_GmshASCII   = MMG5_FMT_GmshASCII,  /*!< ASCII Gmsh */
  PMMG_FMT_GmshBinary  = MMG5_FMT_GmshBinary, /*!< Binary Gmsh */
  PMMG_FMT_VtkPvtp     = MMG5_FMT_VtkPvtp,    /*!< VTK pvtp */
  PMMG_FMT_VtkPvtu     = MMG5_FMT_VtkPvtu,    /*!< VTK pvtu */
  PMMG_FMT_VtkVtu      = MMG5_FMT_VtkVtu,     /*!< VTK vtu */
  PMMG_FMT_VtkVtp      = MMG5_FMT_VtkVtp,     /*!< VTK vtp */
  PMMG_FMT_VtkVtk      = MMG5_FMT_VtkVtk,     /*!< VTK vtk */
  PMMG_FMT_Tetgen      = MMG5_FMT_Tetgen,     /*!< Tetgen or Triangle */
  PMMG_FMT_Centralized,                       /*!< Centralized Setters/Getters */
  PMMG_FMT_Distributed,                       /*!< Distributed Setters/Getters */
  PMMG_FMT_DistributedMeditASCII,             /*!< Distributed ASCII Medit (.mesh) */
  PMMG_FMT_DistributedMeditBinary,            /*!< Distributed Binary Medit (.meshb) */
  PMMG_FMT_Unknown,                           /*!< Unrecognized */
};

/**< Subgroups target size for a fast remeshing step */
static const int PMMG_REMESHER_TARGET_MESH_SIZE = -30000000;

/**< Subgroups target size for a fast remeshing step */
static const int PMMG_REMESHER_NGRPS_MAX = 100;

/**< Number of metis node per mmg mesh... to test*/
static const int PMMG_RATIO_MMG_METIS = -100;

/**< Subgroups target size for a fast remeshing step */
static const int PMMG_REDISTR_NGRPS_MAX = 1000;

/**< Subgroups minimum size to try to avoid empty partitions */
static const int PMMG_REDISTR_NELEM_MIN = 6;

/**< Allowed imbalance ratio between current and demanded groups size */
static const double PMMG_GRPS_RATIO = 2.0;

/**< Number of elements layers for interface displacement */
static const int PMMG_MVIFCS_NLAYERS = 2;

/**
 * \param parmesh pointer toward a parmesh structure
 * \param val     exit value
 *
 * Controlled parmmg termination:
 *   Deallocate parmesh struct and its allocated members
 *   If this is an unsuccessful exit call abort to cancel any remaining processes
 *   Call MPI_Finalize / exit
 */

#define PMMG_RETURN_AND_FREE(parmesh,val) do                            \
  {                                                                     \
                                                                        \
    if ( !PMMG_Free_all( PMMG_ARG_start,                                \
                         PMMG_ARG_ppParMesh,&parmesh,                   \
                         PMMG_ARG_end) ) {                              \
      fprintf(stderr,"  ## Warning: unable to clean the parmmg memory.\n" \
              " Possible memory leak.\n");                              \
    }                                                                   \
                                                                        \
    MPI_Finalize();                                                     \
    return(val);                                                        \
                                                                        \
  } while(0)

/**
 * Clean the mesh, the metric and the solutions and return \a val.
 */
#define PMMG_CLEAN_AND_RETURN(parmesh,val)do                            \
  {                                                                     \
    int kgrp, ksol;                                                     \
                                                                        \
    for ( kgrp=0; kgrp<parmesh->ngrp; ++kgrp ) {                        \
      if ( parmesh->listgrp[kgrp].mesh ) {                              \
        parmesh->listgrp[kgrp].mesh->npi = parmesh->listgrp[kgrp].mesh->np; \
        parmesh->listgrp[kgrp].mesh->nti = parmesh->listgrp[kgrp].mesh->nt; \
        parmesh->listgrp[kgrp].mesh->nai = parmesh->listgrp[kgrp].mesh->na; \
        parmesh->listgrp[kgrp].mesh->nei = parmesh->listgrp[kgrp].mesh->ne; \
      }                                                                 \
                                                                        \
      if ( parmesh->listgrp[kgrp].met )                                 \
        parmesh->listgrp[kgrp].met->npi  = parmesh->listgrp[kgrp].met->np; \
                                                                        \
      if ( parmesh->listgrp[kgrp].mesh ) {                              \
        for ( ksol=0; ksol<parmesh->listgrp[kgrp].mesh->nsols; ++ksol ) { \
          parmesh->listgrp[kgrp].field[ksol].npi  = parmesh->listgrp[kgrp].field[ksol].np; \
        }                                                               \
      }                                                                 \
    }                                                                   \
                                                                        \
    return val;                                                         \
                                                                        \
  }while(0)


#define ERROR_AT(msg1,msg2)                                          \
  fprintf( stderr, msg1 msg2 " function: %s, file: %s, line: %d \n", \
           __func__, __FILE__, __LINE__ )

#define MEM_CHK_AVAIL(mesh,bytes,msg) do {                            \
  if ( (mesh)->memCur + (bytes) > (mesh)->memMax ) {                  \
    ERROR_AT(msg," Exceeded max memory allowed: ");      \
    stat = PMMG_FAILURE;                                              \
  } else if ( (mesh)->memCur + (bytes) < 0  ) {                       \
    ERROR_AT(msg," Tried to free more mem than allocated: " );        \
    stat = PMMG_SUCCESS;                                              \
  }                                                                   \
  else {                                                              \
    stat = PMMG_SUCCESS;                                              \
  } } while(0)

#define PMMG_DEL_MEM(mesh,ptr,type,msg) do {                \
    size_t size_to_free;                                    \
                                                            \
    if ( ptr ) {                                            \
      size_to_free = myfree( ptr );                         \
      assert ( (mesh)->memCur >= size_to_free );            \
      (mesh)->memCur -= size_to_free;                       \
      (ptr) = NULL;                                         \
    }                                                       \
  } while(0)

#define PMMG_MALLOC(mesh,ptr,size,type,msg,on_failure) do { \
  int    stat = PMMG_SUCCESS;                               \
  size_t size_to_allocate;                                  \
                                                            \
  (ptr) = NULL;                                             \
  if ( (size) != 0 ) {                                      \
    size_to_allocate = (size)*sizeof(type);                 \
    MEM_CHK_AVAIL(mesh,size_to_allocate,msg );              \
    if ( stat == PMMG_SUCCESS ) {                           \
      (ptr) = (type*)mymalloc( size_to_allocate );          \
      if ( (ptr) == NULL ) {                                \
        ERROR_AT( msg, " malloc failed: " );                \
        on_failure;                                         \
      } else {                                              \
        (mesh)->memCur += size_to_allocate;                 \
        stat = PMMG_SUCCESS;                                \
      }                                                     \
    } else {                                                \
      on_failure;                                           \
    }                                                       \
  } } while(0)

#define PMMG_CALLOC(mesh,ptr,size,type,msg,on_failure) do { \
  int    stat = PMMG_SUCCESS;                               \
  size_t size_to_allocate;                                  \
                                                            \
  (ptr) = NULL;                                             \
  if ( (size) != 0 ) {                                      \
    size_to_allocate = (size)*sizeof(type);                 \
    MEM_CHK_AVAIL(mesh,size_to_allocate,msg);               \
    if ( stat == PMMG_SUCCESS ) {                           \
      (ptr) = (type*)mycalloc( (size), sizeof(type) );      \
      if ( (ptr) == NULL ) {                                \
        ERROR_AT(msg," calloc failed: ");                   \
        on_failure;                                         \
      } else {                                              \
        (mesh)->memCur += size_to_allocate;                 \
      }                                                     \
    } else {                                                \
      on_failure;                                           \
    }                                                       \
  } } while(0)

#define PMMG_REALLOC(mesh,ptr,newsize,oldsize,type,msg,on_failure) do { \
  int    stat = PMMG_SUCCESS;                                           \
  size_t size_to_allocate,size_to_add,size_to_increase;                 \
  type*  tmp;                                                           \
                                                                        \
  if ( (ptr) == NULL ) {                                                \
    assert(((oldsize)==0) && "NULL pointer pointing to non 0 sized memory?"); \
    PMMG_MALLOC(mesh,ptr,(newsize),type,msg,on_failure);                \
  } else if ((newsize)==0) {                                            \
    PMMG_DEL_MEM(mesh,ptr,type,msg);                                    \
  } else if ((newsize) < (oldsize)) {                                   \
    size_to_allocate = (newsize)*sizeof(type);                          \
    tmp = (type *)myrealloc((ptr),size_to_allocate,                     \
                            (oldsize)*sizeof(type));                    \
    if ( tmp == NULL ) {                                                \
      ERROR_AT(msg," Realloc failed: ");                                \
      PMMG_DEL_MEM(mesh,ptr,type,msg);                                  \
      on_failure;                                                       \
    } else {                                                            \
      (ptr) = tmp;                                                      \
      (mesh)->memCur -= (((oldsize)*sizeof(type))-size_to_allocate);    \
    }                                                                   \
  } else if ((newsize) > (oldsize)) {                                   \
    size_to_add = ((newsize)-(oldsize))*sizeof(type);                   \
    size_to_allocate = (newsize)*sizeof(type);                          \
    size_to_increase = (oldsize)*sizeof(type);                          \
                                                                        \
    MEM_CHK_AVAIL(mesh,size_to_add,msg);                                \
    if ( stat == PMMG_SUCCESS ) {                                       \
      tmp = (type *)myrealloc((ptr),size_to_allocate,size_to_increase); \
      if ( tmp == NULL ) {                                              \
        ERROR_AT(msg, " Realloc failed: " );                            \
        PMMG_DEL_MEM(mesh,ptr,type,msg);                                \
        on_failure;                                                     \
      } else {                                                          \
        (ptr) = tmp;                                                    \
        (mesh)->memCur += ( size_to_add );                              \
      }                                                                 \
    }                                                                   \
    else {                                                              \
      on_failure;                                                       \
    }                                                                   \
  }                                                                     \
  } while(0)

#define PMMG_RECALLOC(mesh,ptr,newsize,oldsize,type,msg,on_failure) do { \
    int my_stat = PMMG_SUCCESS;                                         \
                                                                        \
    PMMG_REALLOC(mesh,ptr,newsize,oldsize,type,msg,my_stat=PMMG_FAILURE;on_failure;); \
    if ( (my_stat == PMMG_SUCCESS ) && ((newsize) > (oldsize)) ) {      \
      memset( (ptr) + oldsize, 0, ((size_t)((newsize)-(oldsize)))*sizeof(type)); \
    }                                                                   \
  } while(0)


/**
 * \param mesh pointer toward a mesh structure
 *
 * Set memMax to memCur. */
#define PMMG_FIT_MEM(mesh) do {                                         \
    mesh->memMax = mesh->memCur;                                        \
  } while(0)

/**
 * \param parmesh pointer toward a parmesh structure
 *
 * Set memMax to memCur for all meshes. */
#define PMMG_FIT_MEM_MESHES(parmesh) do {                               \
    for( igrp = 0; igrp < parmesh->ngrp; igrp++ ) {                     \
      PMMG_FIT_MEM(parmesh->listgrp[igrp].mesh);                        \
    }                                                                   \
    if( parmesh->old_listgrp ) {                                        \
      for( igrp = 0; igrp < parmesh->nold_grp; igrp++ ) {               \
        if( !parmesh->old_listgrp[igrp].mesh ) continue;                \
        PMMG_FIT_MEM(parmesh->old_listgrp[igrp].mesh);                  \
      }                                                                 \
    }                                                                   \
  } while(0)

/**
 * \param parmesh pointer toward a parmesh structure
 * \param mesh pointer toward a mesh structure
 *
 * Take from parmesh the memory used by mesh pointers allocation. */
#define PMMG_GHOSTMEM_INIT(parmesh,mesh) do {                              \
    PMMG_FIT_MEM(mesh);                                                    \
    if( mesh->memCur >= parmesh->memMax ) {                                \
      fprintf(stderr,"\n  ## Error: %s: not enough memory.\n"              \
              "     Allowed: %zu\n"                                        \
              "     Current: %zu\n",__func__,parmesh->memMax,mesh->memCur);\
      assert(0);                                                           \
      return 0;                                                            \
    } else {                                                               \
      parmesh->memMax -= mesh->memCur;                                     \
    }                                                                      \
  } while(0)

/**
 * \param parmesh pointer toward a parmesh structure
 * \param mesh pointer toward a mesh structure
 *
 * Give to parmesh the memory freed by mesh pointers deallocation. */
#define PMMG_GHOSTMEM_FREE(parmesh,mesh) do {                              \
    parmesh->memMax += mesh->memCur;                                       \
  } while(0)

/**
 * \param parmesh pointer toward a parmesh structure
 * \param memUsed amount of memory used
 *
 * Global memory count. */
#define PMMG_COMPUTE_USEDMEM(parmesh,memUsed) do {                      \
    int        igrp;                                                    \
                                                                        \
    memUsed = parmesh->memCur;                                          \
    for( igrp = 0; igrp < parmesh->ngrp; igrp++ ) {                     \
      if( !parmesh->listgrp[igrp].mesh ) continue;                      \
      memUsed += parmesh->listgrp[igrp].mesh->memCur;                   \
    }                                                                   \
    if( parmesh->old_listgrp ) {                                        \
      for( igrp = 0; igrp < parmesh->nold_grp; igrp++ ) {               \
        if( !parmesh->old_listgrp[igrp].mesh ) continue;                \
        memUsed += parmesh->old_listgrp[igrp].mesh->memCur;             \
      }                                                                 \
    }                                                                   \
  } while(0)

/**
 * \param parmesh pointer toward a parmesh structure
 * \param grps pointer toward external groups
 * \param ngrp number of external groups
 * \param memUsed amount of memory used
 *
 * Global memory count, taking into account grous potentially not listed in the
 * parmesh structure.
 * The implementation of the function is transparent to groups that are already
 * listed in the parmesh (i.e. they are counted only once). */
#define PMMG_COMPUTE_USEDMEM_EXT(parmesh,grps,ngrp,memUsed) do {        \
    int i;                                                              \
                                                                        \
    PMMG_COMPUTE_USEDMEM(parmesh,memUsed);                              \
    if( grps && (grps != parmesh->listgrp) ) {                          \
      for( i = 0; i < ngrp; i++ ) {                                     \
        if( !(grps)[i].mesh ) continue;                                 \
        memUsed += (grps)[i].mesh->memCur;                              \
      }                                                                 \
    }                                                                   \
  } while(0)

/**
 * \param parmesh pointer toward a parmesh structure
 * \param mesh pointer toward a mesh structure
 *
 * Assert the global memory count. */
#ifndef NDEBUG
#define PMMG_ASSERT_MEM(parmesh,meshTaker) do {                         \
    size_t memAv,memUsed,memSum;                                        \
                                                                        \
    memAv = meshTaker->memMax-meshTaker->memCur;                        \
    PMMG_COMPUTE_USEDMEM(parmesh,memUsed);                              \
    memSum = memUsed + memAv;                                           \
    if( memSum != parmesh->memGloMax ) {                                \
      fprintf(stderr,"\n  ## Error: %s: memory count mismatch.\n"       \
              "     Total:          %zu -- Used %zu, available %zu\n"   \
              "     Used+available: %zu\n",__func__,                    \
              parmesh->memGloMax,memUsed,memAv,memSum);                 \
      assert(0);                                                        \
    } \
  } while(0)
#else
#define PMMG_ASSERT_MEM(parmesh,meshTaker) do {} while(0)
#endif

/**
 * \param parmesh pointer toward a parmesh structure
 * \param grps pointer toward external groups
 * \param ngrp number of external groups
 * \param mesh pointer toward a mesh structure
 *
 * Assert the global memory count, taking into account grous not listed in the
 * parmesh structure. */
#ifndef NDEBUG
#define PMMG_ASSERT_MEM_EXT(parmesh,grps,ngrp,meshTaker) do {           \
    size_t memAv,memUsed,memSum;                                        \
                                                                        \
    memAv = meshTaker->memMax-meshTaker->memCur;                        \
    PMMG_COMPUTE_USEDMEM_EXT(parmesh,grps,ngrp,memUsed);                \
    memSum = memUsed + memAv;                                           \
    if( memSum != parmesh->memGloMax ) {                                \
      fprintf(stderr,"\n  ## Error: %s: memory count mismatch.\n"       \
              "     Total:          %zu -- Used %zu, available %zu\n"   \
              "     Used+available: %zu\n",__func__,                    \
              parmesh->memGloMax,memUsed,memAv,memSum);                 \
      assert(0);                                                        \
    } \
  } while(0)
#else
#define PMMG_ASSERT_MEM_EXT(parmesh,grps,ngrp,meshTaker) do {} while(0)
#endif

/**
 * \param parmesh pointer toward a parmesh structure
 *
 * Set memMax to memCur for every group mesh, compute the available memory and
 * give it to the parmesh
 *
 */
#define PMMG_TRANSFER_AVMEM_TO_PARMESH(parmesh) do {                     \
    size_t memUsed,memAv;                                                \
    int    igrp;                                                         \
                                                                         \
    PMMG_COMPUTE_USEDMEM(parmesh,memUsed);                               \
    if( memUsed >= parmesh->memGloMax ) {                                \
      fprintf(stderr,"\n  ## Error: %s: not enough memory.\n"            \
              "     Maximum: %zu\n"                                      \
              "     Used:    %zu\n",__func__,parmesh->memGloMax,memUsed);\
      assert(0);                                                         \
      return 0;                                                          \
    } else {                                                             \
      memAv = parmesh->memGloMax-memUsed;                                \
      parmesh->memMax = parmesh->memCur+memAv;                           \
                                                                         \
      PMMG_FIT_MEM_MESHES(parmesh);                                      \
    }                                                                    \
  } while(0)

/**
 * \param parmesh pointer toward a parmesh structure
 *
 * Set memMax to memCur for the parmesh, compute the available memory and
 * repartite it to the mesh
 *
 */
#define PMMG_TRANSFER_AVMEM_TO_MESHES(parmesh) do {                      \
    size_t memUsed,memAv;                                                \
    int    igrp;                                                         \
                                                                         \
    PMMG_COMPUTE_USEDMEM(parmesh,memUsed);                               \
    if( memUsed >= parmesh->memGloMax ) {                                \
      fprintf(stderr,"\n  ## Error: %s: not enough memory.\n"            \
              "     Maximum: %zu\n"                                      \
              "     Used:    %zu\n",__func__,parmesh->memGloMax,memUsed);\
      return 0;                                                          \
    } else {                                                             \
      memAv = parmesh->memGloMax-memUsed;                                \
      memAv /= parmesh->ngrp;                                            \
                                                                         \
      PMMG_FIT_MEM(parmesh);                                             \
      PMMG_FIT_MEM_MESHES(parmesh);                                      \
                                                                         \
      for(  igrp = 0; igrp < parmesh->ngrp; igrp++ )                     \
        parmesh->listgrp[igrp].mesh->memMax += memAv;                    \
    }                                                                    \
  } while(0)

/**
 * \param meshDonor pointer toward the donor mesh
 * \param meshTaker pointer toward the taker mesh
 *
 * Transfer available memory from mesh donor to mesh taker.
 *
 */
#define PMMG_TRANSFER_AVMEM_FROM_MESH_TO_MESH(meshDonor,meshTaker) do {  \
    size_t memAv;                                                        \
                                                                         \
    if( meshDonor->memCur >= meshDonor->memMax ) {                       \
      fprintf(stderr,"\n  ## Error: %s: not enough memory.\n"            \
              "     Allowed: %zu\n"                                      \
              "     Current: %zu\n",__func__,meshDonor->memMax,meshDonor->memCur);\
      assert(0);                                                         \
      return 0;                                                          \
    } else {                                                             \
      memAv = meshDonor->memMax-meshDonor->memCur;                       \
                                                                         \
      meshDonor->memMax = meshDonor->memCur;                             \
      if( meshTaker->memMax != meshTaker->memCur ) {                     \
        fprintf(stderr,"\n  ## Error: %s: taker memory not fitted.\n"    \
                "     Max: %zu, current %zu\n",__func__,                 \
                meshTaker->memMax,meshTaker->memCur);                    \
        assert(0);                                                       \
        return 0;                                                        \
      } else {                                                           \
        meshTaker->memMax = meshTaker->memCur+memAv;                     \
      }                                                                  \
    }                                                                    \
  } while(0)


/**
 * \param parmesh pointer toward a parmesh structure
 * \param mesh pointer toward a mesh structure
 *
 * Limit parmesh->memMax to the currently used memory, update the value of the
 * available memory by removing the amount of memory that has been allocated by
 * the parmesh, and give the available memory (previously given to the parmesh)
 * to the group mesh structure. */
#define PMMG_TRANSFER_AVMEM_FROM_PARMESH_TO_MESH(parmesh,mesh) do {     \
                                                                        \
    if( parmesh->memGloMax ) { /* check on the arguments order */       \
      PMMG_TRANSFER_AVMEM_FROM_MESH_TO_MESH(parmesh,mesh);              \
      PMMG_ASSERT_MEM(parmesh,mesh);                                    \
    }                                                                   \
  } while(0)

/**
 * \param parmesh pointer toward a parmesh structure
 * \param grps pointer toward external groups
 * \param ngrp number of external groups
 * \param mesh pointer toward a mesh structure
 *
 * Limit parmesh->memMax to the currently used memory, update the value of the
 * available memory by removing the amount of memory that has been allocated by
 * the parmesh, and give the available memory (previously given to the parmesh)
 * to the group mesh structure.
 * Also take into account groups not listed in the parmesh structure. */
#define PMMG_TRANSFER_AVMEM_FROM_PARMESH_TO_MESH_EXT(parmesh,grps,ngrp,mesh) do { \
                                                                                  \
    PMMG_TRANSFER_AVMEM_FROM_MESH_TO_MESH(parmesh,mesh);                          \
    PMMG_ASSERT_MEM_EXT(parmesh,grps,ngrp,mesh);                                  \
  } while(0)

/**
 * \param parmesh pointer toward a parmesh structure
 * \param grps pointer toward external groups
 * \param ngrp number of external groups
 * \param mesh pointer toward a mesh structure
 *
 * Limit mesh->memMax to the currently used memory, update the value of the
 * available memory by removing the amount of memory that has been allocated by
 * the mesh, and give the available memory (previously given to the mesh)
 * to the parmesh structure. */
#define PMMG_TRANSFER_AVMEM_FROM_MESH_TO_PARMESH(parmesh,mesh) do {     \
                                                                        \
    if( parmesh->memGloMax ) { /* check on the arguments order */       \
      PMMG_TRANSFER_AVMEM_FROM_MESH_TO_MESH(mesh,parmesh);              \
      PMMG_ASSERT_MEM(parmesh,parmesh);                                 \
    }                                                                   \
  } while(0)

/**
 * \param parmesh pointer toward a parmesh structure
 * \param mesh pointer toward a mesh structure
 *
 * Limit mesh->memMax to the currently used memory, update the value of the
 * available memory by removing the amount of memory that has been allocated by
 * the mesh, and give the available memory (previously given to the mesh)
 * to the parmesh structure.
 * Also take into account groups not listed in the parmesh structure. */
#define PMMG_TRANSFER_AVMEM_FROM_MESH_TO_PARMESH_EXT(parmesh,grps,ngrp,mesh) do { \
                                                                                  \
    PMMG_TRANSFER_AVMEM_FROM_MESH_TO_MESH(mesh,parmesh);                          \
    PMMG_ASSERT_MEM_EXT(parmesh,grps,ngrp,parmesh);                               \
  } while(0)


/* Input */
int PMMG_Set_name(PMMG_pParMesh,char **,const char* name,const char* defname);
int PMMG_check_inputData ( PMMG_pParMesh parmesh );
int PMMG_preprocessMesh( PMMG_pParMesh parmesh );
int PMMG_preprocessMesh_distributed( PMMG_pParMesh parmesh );
int PMMG_parsar( int argc, char *argv[], PMMG_pParMesh parmesh );
void PMMG_setfunc( PMMG_pParMesh parmesh );

/* Mesh analysis */
int PMMG_analys_tria(PMMG_pParMesh parmesh,MMG5_pMesh mesh);
int PMMG_analys(PMMG_pParMesh parmesh,MMG5_pMesh mesh);
int PMMG_hashPar( MMG5_pMesh mesh,MMG5_HGeom *pHash );

/* Internal library */
void PMMG_setfunc( PMMG_pParMesh parmesh );
int PMMG_parmmglib1 ( PMMG_pParMesh parmesh );

/* Mesh distrib */
int PMMG_bdryUpdate( MMG5_pMesh mesh );
int PMMG_bcast_mesh ( PMMG_pParMesh parmesh );
int PMMG_partBcast_mesh( PMMG_pParMesh parmesh );
int PMMG_grpSplit_setMeshSize( MMG5_pMesh,int,int,int,int,int );
int PMMG_splitPart_grps( PMMG_pParMesh,int,int,int );
int PMMG_split_grps( PMMG_pParMesh parmesh,int grpIdOld,int ngrp,idx_t *part,int fitMesh );

/* Load Balancing */
int PMMG_interactionMap(PMMG_pParMesh parmesh,int **interactions,int **interaction_map);
int PMMG_transfer_all_grps(PMMG_pParMesh parmesh,idx_t *part,int);
int PMMG_distribute_grps( PMMG_pParMesh parmesh );
int PMMG_loadBalancing( PMMG_pParMesh parmesh );
int PMMG_split_n2mGrps( PMMG_pParMesh,int,int );
double PMMG_computeWgt( MMG5_pMesh mesh,MMG5_pSol met,MMG5_pTetra pt,int ifac );
void PMMG_computeWgt_mesh( MMG5_pMesh mesh,MMG5_pSol met,int tag );

/* Mesh interpolation */
int PMMG_oldGrps_newGroup( PMMG_pParMesh parmesh,int igrp );
int PMMG_oldGrps_fillGroup( PMMG_pParMesh parmesh,int igrp );
int PMMG_update_oldGrps( PMMG_pParMesh parmesh );
int PMMG_interpMetricsAndFields( PMMG_pParMesh parmesh,int* );
int PMMG_copyMetricsAndFields_point( MMG5_pMesh mesh, MMG5_pMesh oldMesh, MMG5_pSol met, MMG5_pSol oldMet, MMG5_pSol,MMG5_pSol, int* permNodGlob,uint8_t);

/* Communicators building and unallocation */
void PMMG_parmesh_int_comm_free( PMMG_pParMesh,PMMG_pInt_comm);
void PMMG_parmesh_ext_comm_free( PMMG_pParMesh,PMMG_pExt_comm,int);
void PMMG_grp_comm_free( PMMG_pParMesh ,int**,int**,int*);
void PMMG_node_comm_free( PMMG_pParMesh );
void PMMG_edge_comm_free( PMMG_pParMesh );

void PMMG_tria2elmFace_flags( PMMG_pParMesh parmesh );
void PMMG_tria2elmFace_coords( PMMG_pParMesh parmesh );
int PMMG_build_nodeCommIndex( PMMG_pParMesh parmesh );
int PMMG_build_faceCommIndex( PMMG_pParMesh parmesh );
int PMMG_build_nodeCommFromFaces( PMMG_pParMesh parmesh );
int PMMG_build_faceCommFromNodes( PMMG_pParMesh parmesh );
int PMMG_build_simpleExtNodeComm( PMMG_pParMesh parmesh );
int PMMG_build_intNodeComm( PMMG_pParMesh parmesh );
int PMMG_build_completeExtNodeComm( PMMG_pParMesh parmesh );
int PMMG_build_edgeComm( PMMG_pParMesh parmesh,MMG5_pMesh mesh,MMG5_HGeom *hpar );

int PMMG_pack_faceCommunicators(PMMG_pParMesh parmesh);
int PMMG_pack_nodeCommunicators(PMMG_pParMesh parmesh);

/* Communicators checks */
int PMMG_check_intFaceComm( PMMG_pParMesh parmesh );
int PMMG_check_extFaceComm( PMMG_pParMesh parmesh );
int PMMG_check_intNodeComm( PMMG_pParMesh parmesh );
int PMMG_check_extNodeComm( PMMG_pParMesh parmesh );
int PMMG_check_extEdgeComm( PMMG_pParMesh parmesh );

/* Tags */
void PMMG_tag_par_node(MMG5_pPoint ppt);
void PMMG_tag_par_edge(MMG5_pxTetra pxt,int j);
void PMMG_tag_par_face(MMG5_pxTetra pxt,int j);
void PMMG_untag_par_node(MMG5_pPoint ppt);
void PMMG_untag_par_edge(MMG5_pxTetra pxt,int j);
void PMMG_untag_par_face(MMG5_pxTetra pxt,int j);
int  PMMG_resetOldTag(PMMG_pParMesh parmesh);
int  PMMG_updateTag(PMMG_pParMesh parmesh);
int  PMMG_parbdySet( PMMG_pParMesh parmesh );

/* Mesh merge */
int PMMG_mergeGrpJinI_interfacePoints_addGrpJ( PMMG_pParMesh,PMMG_pGrp,PMMG_pGrp);
int PMMG_mergeGrps_interfacePoints( PMMG_pParMesh parmesh );
int PMMG_mergeGrpJinI_internalPoints( PMMG_pGrp,PMMG_pGrp grpJ );
int PMMG_mergeGrpJinI_interfaceTetra( PMMG_pParMesh,PMMG_pGrp,PMMG_pGrp );
int PMMG_mergeGrpJinI_internalTetra( PMMG_pGrp,PMMG_pGrp );
int PMMG_merge_grps ( PMMG_pParMesh parmesh,int );

/* Move interfaces */
int PMMG_part_getInterfaces( PMMG_pParMesh parmesh,int *part,int *ngrps,int target );
int PMMG_part_getProcs( PMMG_pParMesh parmesh,int *part );
int PMMG_fix_contiguity( PMMG_pParMesh parmesh,int *counter );
int PMMG_fix_contiguity_centralized( PMMG_pParMesh parmesh,idx_t *part );
int PMMG_fix_contiguity_split( PMMG_pParMesh parmesh,idx_t ngrp,idx_t *part );
int PMMG_part_moveInterfaces( PMMG_pParMesh parmesh,int *vtxdist,int *map,int *base_front );
int PMMG_mark_interfacePoints( PMMG_pParMesh parmesh,MMG5_pMesh mesh,int* vtxdist,int* priorityMap );
int PMMG_init_ifcDirection( PMMG_pParMesh parmesh,int **vtxdist,int **map );
int PMMG_set_ifcDirection( PMMG_pParMesh parmesh,int **vtxdist,int **map );
int PMMG_get_ifcDirection( PMMG_pParMesh parmesh,int *vtxdist,int *map,int color0,int color1 );

/* Packing */
int PMMG_update_node2intPackedTetra( PMMG_pGrp grp );
int PMMG_mark_packedTetra(MMG5_pMesh mesh,int *ne);
int PMMG_update_node2intPackedVertices( PMMG_pGrp grp );
int PMMG_packTetra ( PMMG_pParMesh parmesh, int igrp );

/* Memory */
int  PMMG_link_mesh( MMG5_pMesh mesh );
void PMMG_listgrp_free( PMMG_pParMesh parmesh, PMMG_pGrp *listgrp, int ngrp );
void PMMG_grp_free( PMMG_pParMesh parmesh, PMMG_pGrp grp );
int  PMMG_parmesh_SetMemMax( PMMG_pParMesh parmesh, int percent);
int  PMMG_setMemMax_realloc( MMG5_pMesh,int,int,int,int);
int  PMMG_parmesh_fitMesh( PMMG_pParMesh parmesh, PMMG_pGrp );
int  PMMG_parmesh_updateMemMax( PMMG_pParMesh parmesh,int percent,int fitMesh);
void PMMG_parmesh_SetMemGloMax( PMMG_pParMesh parmesh );
void PMMG_parmesh_Free_Comm( PMMG_pParMesh parmesh );
void PMMG_parmesh_Free_Listgrp( PMMG_pParMesh parmesh );
int  PMMG_clean_emptyMesh( PMMG_pParMesh parmesh, PMMG_pGrp listgrp, int ngrp );
int  PMMG_resize_extComm ( PMMG_pParMesh,PMMG_pExt_comm,int,int* );
int  PMMG_resize_extCommArray ( PMMG_pParMesh,PMMG_pExt_comm*,int,int*);

/* Tools */
int PMMG_copy_mmgInfo ( MMG5_Info *info, MMG5_Info *info_cpy );

/* Quality */
int PMMG_qualhisto( PMMG_pParMesh parmesh,int,int );
int PMMG_prilen( PMMG_pParMesh parmesh,int8_t,int );
int PMMG_tetraQual( PMMG_pParMesh parmesh,int8_t metRidTyp );

/* Variadic_pmmg.c */
int PMMG_Init_parMesh_var_internal(va_list argptr,int callFromC);
int PMMG_Free_all_var(va_list argptr);

const char* PMMG_Get_pmmgArgName(int typArg);


#ifdef __cplusplus
}
#endif

#endif
