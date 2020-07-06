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
 * \file distributemesh.c
 * \brief Distribute the mesh on the processors.
 * \author Cécile Dobrzynski (Bx INP/Inria/UBordeaux)
 * \author Algiane Froehly (Inria/UBordeaux)
 * \version 5
 * \copyright GNU Lesser General Public License.
 * \todo doxygen documentation.
 */
#include "parmmg.h"
#include "metis_pmmg.h"

static inline
int PMMG_create_empty_communicators( PMMG_pParMesh parmesh ) {
  PMMG_pGrp       grp;

  grp    = &parmesh->listgrp[0];

  /** Internal communicators allocation */
  parmesh->next_node_comm = 0;
  parmesh->next_face_comm = 0;
  grp->nitem_int_node_comm = 0;
  grp->nitem_int_face_comm = 0;

  PMMG_CALLOC(parmesh,parmesh->int_node_comm,1,PMMG_Int_comm,
              "allocating int_node_comm",return 0);
  PMMG_CALLOC(parmesh,parmesh->int_face_comm,1,PMMG_Int_comm,
              "allocating int_face_comm",return 0);
  parmesh->int_node_comm->nitem = 0;
  parmesh->int_face_comm->nitem = 0;

  return 1;
}

/**
 * \param parmesh pointer toward a PMMG parmesh structure.
 * \param part pointer toward the metis array containing the partitions.
 *
 * \return 0 (on all procs) if fail, 1 otherwise
 *
 * Delete parts of the mesh not on the processor.
 */
int PMMG_distribute_mesh( PMMG_pParMesh parmesh )
{
  PMMG_pGrp  grp;
  MMG5_pMesh mesh;
  idx_t      *part;
  int        igrp,ier,ieresult;
  size_t     available,oldMemMax;

  ier = 1;

  /* Create empty communicators on all procs */
  if( !PMMG_create_empty_communicators( parmesh ) ) return 0;

  /* There is nothing to distribute on just 1 proc */
  if( parmesh->nprocs == 1 ) return 1;


  /**
   * 1) Proc 0 partitions the mesh.
   */
  if( parmesh->myrank == parmesh->info.root ) {

    grp    = &parmesh->listgrp[0];
    mesh   = grp->mesh;

    /** Call metis for partionning */
    PMMG_CALLOC ( parmesh,part,mesh->ne,idx_t,"allocate metis buffer", ier=5 );

    /* Call metis, or recover a custom partitioning if provided (only to debug
     * the interface displacement, adaptation will be blocked) */
    if( !PMMG_PREDEF_PART ) {
      if ( !PMMG_part_meshElts2metis( parmesh, part, parmesh->nprocs ) ) {
        ier = 5;
      }
      if( !PMMG_fix_contiguity_centralized( parmesh,part ) ) ier = 5;
    } else {
      int k;
      for( k = 1; k <= mesh->ne; k++ ) {
        part[k-1] = mesh->tetra[k].ref;
        mesh->tetra[k].tag |= MG_REQ;
      }
    }

    /* Split grp 0 into (nprocs) groups */
    ier = PMMG_split_grps( parmesh,0,parmesh->nprocs,part,1 );

    /** Check grps contiguity */
    ier = PMMG_checkAndReset_grps_contiguity( parmesh );

    PMMG_DEL_MEM(parmesh,part,idx_t,"deallocate metis buffer");
  }
  /* At this point all communicators have been created and all tags are OK */


  /**
   * 2) Distribute the groups over the processors.
   */
  if( parmesh->myrank != parmesh->info.root ) {
    PMMG_listgrp_free( parmesh, &parmesh->listgrp, parmesh->ngrp );
    parmesh->ngrp = 0;
  }

  /* Create the groups partition array */
  PMMG_CALLOC ( parmesh,part,parmesh->nprocs+1,idx_t,"allocate metis buffer", ier=5 );
  for( igrp = 0; igrp <= parmesh->nprocs; igrp++ )
    part[igrp] = igrp;

  /* Transfer the groups in parallel */
  ier = PMMG_transfer_all_grps(parmesh,part);
  if ( ier <= 0 ) {
    fprintf(stderr,"\n  ## Group distribution problem.\n");
  }

  assert( parmesh->ngrp = 1);
  grp = &parmesh->listgrp[0];
  mesh = grp->mesh;

  PMMG_TRANSFER_AVMEM_TO_PARMESH(parmesh,available,oldMemMax);
  PMMG_TRANSFER_AVMEM_FROM_PMESH_TO_MESH(parmesh,mesh,available,oldMemMax);
  if ( (!mesh->adja) && !MMG3D_hashTetra(mesh,1) ) {
    fprintf(stderr,"\n  ## Error: %s: tetra hashing problem. Exit program.\n",
            __func__);
    return 0;
  }
  PMMG_TRANSFER_AVMEM_FROM_MESH_TO_PMESH(parmesh,mesh,available,oldMemMax);

  /* At this point all communicators have been created and all tags are OK */


  /* Check the communicators */
  assert ( PMMG_check_intNodeComm(parmesh) && "Wrong internal node comm" );
  assert ( PMMG_check_intFaceComm(parmesh) && "Wrong internal face comm" );
  assert ( PMMG_check_extNodeComm(parmesh) && "Wrong external node comm" );
  assert ( PMMG_check_extFaceComm(parmesh) && "Wrong external face comm" );

  /* The part array is deallocated when groups to be sent are merged (do not
   * do it here) */
  ieresult = ier;
  return ieresult==1;
}
