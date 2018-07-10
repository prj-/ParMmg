#include "parmmg.h"
#include <stddef.h>

typedef struct {
  double min;
  int iel, iel_grp, cpu;
} min_iel_t;

typedef struct {
  double min;
  int amin, bmin,grp_min,cpu_min;
  double max;
  int amax, bmax,grp_max,cpu_max;
} min_max_t;

static void PMMG_min_iel_compute( void* in1, void* out1, int *len, MPI_Datatype *dptr )
{
  int i;
  min_iel_t *in;
  min_iel_t *out;

  in = (min_iel_t*) in1;
  out = (min_iel_t*) out1;
  (void)dptr;
  for (int i=0; i<*len; i++) {
    if ( in[ i ].min < out[ i ]. min ) {
      out[ i ].min = in[ i ].min;
      out[ i ].iel = in[ i ].iel;
      out[ i ].iel_grp = in[ i ].iel_grp;
      out[ i ].cpu = in[ i ].cpu;
    }
  }
}

static void PMMG_min_max_compute( void* in1, void* out1, int *len, MPI_Datatype *dptr )
{
  int i;
  min_max_t *in;
  min_max_t *out;

  in = (min_max_t*) in1;
  out = (min_max_t*) out1;
  (void)dptr;
  for ( i = 0; i < *len; i++ ) {
    if ( in[ i ].min < out[ i ].min ) {
      out[ i ].min     = in[ i ].min;
      out[ i ].amin    = in[ i ].amin;
      out[ i ].bmin    = in[ i ].bmin;
      out[ i ].grp_min = in[ i ].grp_min;
      out[ i ].cpu_min = in[ i ].cpu_min;
    }
    if ( in[ i ].max > out[ i ].max ) {
      out[ i ].max     = in[ i ].max;
      out[ i ].amax    = in[ i ].amax;
      out[ i ].bmax    = in[ i ].bmax;
      out[ i ].grp_max = in[ i ].grp_max;
      out[ i ].cpu_max = in[ i ].cpu_max;
    }
  }
}

/**
 * \param parmesh pointer to parmesh structure
 *
 * \return 1 if success, 0 if fail;
 *
 * Print quality histogram among all group meshes and all processors
 */
int PMMG_outqua( PMMG_pParMesh parmesh )
{
  PMMG_pGrp grp;
  int i, j, ier, ieresult, iel_grp;
  int ne, ne_cur, ne_result;
  double max, max_cur, max_result;
  double avg, avg_cur, avg_result;
  double min, min_cur;
  int iel, iel_cur;
  int good, good_cur, good_result;
  int med, med_cur, med_result;
  const int HIS_SIZE = 5;
  int his[ HIS_SIZE ], his_cur[ HIS_SIZE ], his_result[ HIS_SIZE ];
  int nrid, nrid_cur, nrid_result;
  MPI_Op        iel_min_op;
  MPI_Datatype  mpi_iel_min_t;
  MPI_Datatype types[ 4 ] = { MPI_DOUBLE, MPI_INT, MPI_INT, MPI_INT };
  min_iel_t     min_iel, min_iel_result = { DBL_MAX, 0, 0, 0 };
  MPI_Aint disps[ 4 ] = { offsetof( min_iel_t, min ),
                          offsetof( min_iel_t, iel ),
                          offsetof( min_iel_t, iel_grp ),
                          offsetof( min_iel_t, cpu ) };
  int lens[ 4 ] = { 1, 1, 1, 1 };

  ier = 1;

  // Calculate the quality values for local process
  iel_grp = 0;
  ne = 0;
  max = DBL_MIN;
  avg = 0.;
  min = DBL_MAX;
  iel = 0;
  good = 0;
  med = 0;

  for ( i = 0; i < HIS_SIZE; ++i )
    his[ i ] = 0;

  nrid = 0;
  for ( i = 0; i < parmesh->ngrp; ++i ) {
    grp  = &parmesh->listgrp[ i ];
    MMG3D_computeOutqua( grp->mesh, grp->met, &ne_cur, &max_cur, &avg_cur, &min_cur,
                         &iel_cur, &good_cur, &med_cur, his_cur, &nrid_cur );

    ne   += ne_cur;
    avg  += avg_cur;
    med  += med_cur;
    good += good_cur;

    if ( max_cur > max )
      max = max_cur;

    if ( min_cur < min ) {
      min = min_cur;
      iel = iel_cur;
      iel_grp = i;
    }

    for ( j = 0; j < HIS_SIZE; ++j )
      his[ j ] += his_cur[ j ];

    nrid += nrid_cur;
  }

  // Calculate the quality values for all processes
  MPI_Reduce( &ne, &ne_result, 1, MPI_INT, MPI_SUM, 0, parmesh->comm );
  MPI_Reduce( &avg, &avg_result, 1, MPI_DOUBLE, MPI_SUM, 0, parmesh->comm );
  MPI_Reduce( &med, &med_result, 1, MPI_INT, MPI_SUM, 0, parmesh->comm );
  MPI_Reduce( &good, &good_result, 1, MPI_INT, MPI_SUM, 0, parmesh->comm );
  MPI_Reduce( &max, &max_result, 1, MPI_DOUBLE, MPI_MAX, 0, parmesh->comm );

  MPI_Type_create_struct( 4, lens, disps, types, &mpi_iel_min_t );
  MPI_Type_commit( &mpi_iel_min_t );
  MPI_Op_create( PMMG_min_iel_compute, 1, &iel_min_op );
  min_iel.min = min;
  min_iel.iel = iel;
  min_iel.iel_grp = iel_grp;
  min_iel.cpu = parmesh->myrank;
  MPI_Reduce( &min_iel, &min_iel_result, 1, mpi_iel_min_t, iel_min_op, 0, parmesh->comm );
  MPI_Op_free( &iel_min_op );

  MPI_Reduce( his, his_result, HIS_SIZE, MPI_INT, MPI_SUM, 0, parmesh->comm );
  MPI_Reduce( &nrid, &nrid_result, 1, MPI_INT, MPI_SUM, 0, parmesh->comm );

  if ( parmesh->myrank == 0 ) {

    fprintf(stdout,"\n  -- PARALLEL MESH QUALITY");

    grp = &parmesh->listgrp[ 0 ];
#warning to change with parmesh->imprim once the leadbalanding branch will be merged
    if ( grp->mesh->info.imprim )
      fprintf( stdout," (LES)" );
    fprintf( stdout, "  %d\n", ne_result );

    fprintf( stdout, "     BEST   %8.6f  AVRG.   %8.6f  WRST.   %8.6f (",
             max_result, avg_result / ne_result, min_iel_result.min);

    if ( parmesh->ngrp>1 )
      fprintf( stdout, "GROUP %d - ",min_iel_result.iel_grp);

    if ( parmesh->nprocs>1 )
      fprintf( stdout, "PROC %d - ",min_iel_result.cpu);

    fprintf( stdout,"ELT %d)\n", min_iel_result.iel );

    if ( !MMG3D_displayQualHisto_internal( ne_result, max_result, avg_result,
                                          min_iel_result.min, min_iel_result.iel,
                                          good_result, med_result, his_result,
                                          nrid_result,grp->mesh->info.optimLES,
                                          grp->mesh->info.imprim ) )
      ier = 0;
  }

  MPI_Allreduce( &ier, &ieresult, 1, MPI_INT, MPI_MIN, parmesh->comm );

  return ieresult;
}

/**
 * \param parmesh pointer toward a parmesh structure
 * \param metRidTyp Type of storage of ridges metrics: 0 for classic storage,
 * 1 for special storage.
 *
 * \return 1 if success, 0 if fail;
 *
 * Print edge length quality histogram among all group meshes and all processors
 */
int PMMG_prilen( PMMG_pParMesh parmesh,char metRidTyp)
{
  PMMG_pGrp grp;
  double *bd;
  double avlen, avlen_cur, avlen_result;
  double lmin, lmin_cur, lmax, lmax_cur;
  int ned, ned_cur, ned_result;
  int amin, amin_cur, bmin, bmin_cur, amax, amax_cur, bmax, bmax_cur;
  int nullEdge, nullEdge_cur, nullEdge_result;
  int hl[ 9 ], hl_cur[ 9 ], hl_result[ 9 ];
  int i,grp_min,grp_max;
  MPI_Op        min_max_op;
  MPI_Datatype  mpi_min_max_t;
  MPI_Datatype types[ 10 ] = { MPI_DOUBLE, MPI_INT, MPI_INT, MPI_INT, MPI_INT,
                              MPI_DOUBLE, MPI_INT, MPI_INT, MPI_INT, MPI_INT};
  min_max_t min_max, min_max_result = { DBL_MAX, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  MPI_Aint disps[ 10 ] = { offsetof( min_max_t, min ),
                          offsetof( min_max_t, amin ),
                          offsetof( min_max_t, bmin ),
                          offsetof( min_max_t, grp_min ),
                          offsetof( min_max_t, cpu_min ),
                          offsetof( min_max_t, max ),
                          offsetof( min_max_t, amax ),
                          offsetof( min_max_t, bmax ),
                          offsetof( min_max_t, grp_max ),
                          offsetof( min_max_t, cpu_max )};
  int lens[ 10 ] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };


  nullEdge = 0;
  avlen = 0;
  ned = 0;
  lmin = DBL_MAX;
  lmax = 0;

  for ( i = 0; i < 9; ++i )
    hl[ i ] = 0;

  grp_min = grp_max = 0;

  for ( i = 0; i < parmesh->ngrp; ++i ) {
    grp  = &parmesh->listgrp[ i ];
    MMG3D_computePrilen( grp->mesh, grp->met, &avlen_cur, &lmin_cur, &lmax_cur,
                         &ned_cur, &amin_cur, &bmin_cur, &amax_cur, &bmax_cur,
                         &nullEdge_cur, metRidTyp, &bd, hl );

    nullEdge += nullEdge_cur;
    avlen += avlen_cur;
    ned += ned_cur;
    if ( lmin_cur < lmin ) {
      lmin    = lmin_cur;
      amin    = amin_cur;
      bmin    = bmin_cur;
      grp_min = i;
    }
    if ( lmax_cur > lmax ) {
      lmax    = lmax_cur;
      amax    = amax_cur;
      bmax    = bmax_cur;
      grp_max = i;
    }
    for ( i = 0; i < 9; ++i )
      hl[ i ] += hl_cur[ i ];
  }

  MPI_Reduce( &nullEdge, &nullEdge_result, 1, MPI_INT, MPI_SUM, 0, parmesh->comm );
  MPI_Reduce( &avlen, &avlen_result, 1, MPI_DOUBLE, MPI_SUM, 0, parmesh->comm );
  MPI_Reduce( &ned, &ned_result, 1, MPI_INT, MPI_SUM, 0, parmesh->comm );
  for ( i = 0; i < 9; ++i )
    MPI_Reduce( hl + i, hl_result + i, 1, MPI_INT, MPI_SUM, 0, parmesh->comm );

  MPI_Type_create_struct( 10, lens, disps, types, &mpi_min_max_t );
  MPI_Type_commit( &mpi_min_max_t );
  MPI_Op_create( PMMG_min_max_compute, 1, &min_max_op );
  min_max.min     = lmin;
  min_max.amin    = amin;
  min_max.bmin    = bmin;
  min_max.grp_min = grp_min;
  min_max.cpu_min = parmesh->myrank;

  min_max.max     = lmax;
  min_max.amax    = amax;
  min_max.bmax    = bmax;
  min_max.grp_max = grp_max;
  min_max.cpu_max = parmesh->myrank;

  MPI_Reduce( &min_max, &min_max_result, 1, mpi_min_max_t, min_max_op, 0, parmesh->comm );
  MPI_Op_free( &min_max_op );

  if ( parmesh->myrank == 0 ) {
    avlen_result = avlen_result/(double)ned_result;

    fprintf(stdout,"\n  -- RESULTING EDGE LENGTHS  %d\n",ned);
    fprintf(stdout,"     AVERAGE LENGTH         %12.4f\n",avlen_result);

    fprintf(stdout,"     SMALLEST EDGE LENGTH   %12.4f   ",min_max_result.min);
    if ( parmesh->ngrp>1 )
      fprintf( stdout, "GROUP %d - ",min_max_result.grp_min);

    if ( parmesh->nprocs>1 )
      fprintf( stdout, "PROC %d - ",min_max_result.cpu_min);

    fprintf( stdout,"EDGE %6d %6d\n",min_max_result.amin,min_max_result.bmin);

    fprintf(stdout,"     LARGEST EDGE LENGTH   %12.4f   ",min_max_result.max);
    if ( parmesh->ngrp>1 )
      fprintf( stdout, "GROUP %d - ",min_max_result.grp_max);

    if ( parmesh->nprocs>1 )
      fprintf( stdout, "PROC %d - ",min_max_result.cpu_max);

    fprintf( stdout,"EDGE %6d %6d\n",min_max_result.amax,min_max_result.bmax);

#warning to change with parmesh->imprim once the leadbalanding branch will be merged
    _MMG5_displayLengthHisto_internal( parmesh->listgrp[ 0 ].mesh, ned_result,
                                       min_max_result.amin, min_max_result.bmin,
                                       min_max_result.min, min_max_result.amax,
                                       min_max_result.bmax, min_max_result.max,
                                       nullEdge_result, bd, hl_result, 1,
                                       parmesh->listgrp[0].mesh->info.imprim);
  }

  return 1;
}

