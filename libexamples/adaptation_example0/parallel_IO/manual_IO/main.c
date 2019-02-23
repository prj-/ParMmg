/**
 * Example of use of the parmmg library (basic use of mesh adaptation).
 *
 * This example show how to set and get the mesh, metric and solutions using the
 * ParMmg setters and getters. Depending of the command line option "opt", the
 * values are setted entities by entities or the entire value array is directly
 * setted.
 *
 * \author Luca Cirrottola (Inria)
 * \author Algiane Froehly (InriaSoft)
 * \version 1
 * \copyright GNU Lesser General Public License.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <float.h>

/** Include the parmmg library hader file */
// if the header file is in the "include" directory
#include "libparmmg.h"
// if the header file is in "include/parmmg"
//#include "parmmg/libparmmg.h"

#define MAX2(a,b)      (((a) > (b)) ? (a) : (b))
#define MAX4(a,b,c,d)  (((MAX2(a,b)) > (MAX2(c,d))) ? (MAX2(a,b)) : (MAX2(c,d)))

void append_ifaces(int shift, int nifc, int *inv_pmask,
                   int *tvert, int *tref, int *ticf,int *tifc_ref) {
  int k,d;
  for( k=0; k<nifc; k++ ) {
    for( d=0; d<3; d++ )
      tvert[3*(k+shift)+d] = inv_pmask[ticf[3*k+d]];
    tref[k+shift] = tifc_ref[k];
  }
}

void get_local_mesh(int np, int ne, int nt, int *pmask, int *inv_pmask,
                    int *emask, int *tmask, int *inv_tmask,
                    double *pcoor, double *pcoor_all, int *pref, int *pref_all,
                    int *evert, int *evert_all, int *eref, int *eref_all,
                    int *tvert, int *tvert_all, int *tref, int *tref_all,
                    double *met, double *met_all,int ncomm,
                    int *ntifc, int **ifc_tria_loc, int **ifc_tria_glob,
                    int *npifc, int **ifc_nodes_loc, int **ifc_nodes_glob) {

  int k,d,icomm;

  for( k=0; k<np; k++ ) {
    inv_pmask[pmask[k]] = k;
    for( d=0; d<3; d++ )
      pcoor[3*k+d] = pcoor_all[3*pmask[k]+d];
    pref[k] = pref_all[pmask[k]];
  }
  for( k=0; k<ne; k++ ) {
    for( d=0; d<4; d++ )
      evert[4*k+d] = inv_pmask[evert_all[4*emask[k]+d]-1]+1;
    eref[k] = eref_all[emask[k]];
  }
  for( k=0; k<nt; k++ ) {
    inv_tmask[tmask[k]] = k;
    for( d=0; d<3; d++ )
      tvert[3*k+d] = inv_pmask[tvert_all[3*tmask[k]+d]-1]+1;
    tref[k] = tref_all[tmask[k]];
  }
  for( k=0; k<np; k++ ) {
    met[k] = met_all[pmask[k]];
  }
  for( icomm=0; icomm<ncomm; icomm++ ) {
    for( k=0; k<ntifc[icomm]; k++ ) {
      ifc_tria_loc[icomm][k] = inv_tmask[ifc_tria_glob[icomm][k]-1]+1;
    }
    for( k=0; k<npifc[icomm]; k++ ) {
      ifc_nodes_loc[icomm][k] = inv_pmask[ifc_nodes_glob[icomm][k]-1]+1;
    }
  }
}

int main(int argc,char *argv[]) {
  PMMG_pParMesh   parmesh;
  int             ier,ierlib,rank,k,opt,API_mode,i,d,icomm;
  char            *metname,*solname,*fileout,*metout,*solout,*tmp;
  FILE            *inm;
  int             pos,nreq,nc,nr;


  MPI_Init( &argc, &argv );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  if ( !rank ) fprintf(stdout,"  -- TEST PARMMGLIB \n");

  solname = NULL;
  metname = NULL;
  solout  = NULL;
  metout  = NULL;
  tmp     = NULL;

  if ( (argc!=4) && !rank ) {
    printf(" Usage: %s fileout io_option\n",argv[0]);
    printf("     io_option = 0 to Get/Set the mesh/metric/solution by array\n");
    printf("     io_option = 1 to Get/Set the mesh/metric/solution vertex by vertex\n");
    return 1;
  }

  fileout = (char *) calloc(strlen(argv[1]) + 6 + 4, sizeof(char));
  if ( fileout == NULL ) {
    perror("  ## Memory problem: calloc");
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }
  strcpy(fileout,argv[1]);
  sprintf(fileout, "%s-P%02d", fileout, rank );
  strcat(fileout,".mesh");

  metout = (char *) calloc(strlen(argv[1]) + 9 + 4, sizeof(char));
  if ( metout == NULL ) {
    perror("  ## Memory problem: calloc");
    exit(EXIT_FAILURE);
  }
  strcpy(metout,argv[1]);
  sprintf(metout, "%s-P%02d", metout, rank );
  strcat(metout,"-met.sol");

  solout = (char *) calloc(strlen(argv[1]) + 13 + 4, sizeof(char));
  if ( solout == NULL ) {
    perror("  ## Memory problem: calloc");
    exit(EXIT_FAILURE);
  }
  strcpy(solout,argv[1]);
  sprintf(solout, "%s-P%02d", solout, rank );
  strcat(solout,"-solphys.sol");

  opt      = atoi(argv[2]);
  API_mode = atoi(argv[3]);

  /** ------------------------------ STEP   I -------------------------- */
  /** 1) Initialisation of th parmesh structures */
  /* args of InitMesh:
   * PMMG_ARG_start: we start to give the args of a variadic func
   * PMMG_ARG_ppParMesh: next arg will be a pointer over a PMMG_pParMesh
   * &parmesh: pointer toward your PMMG_pParMesh
   * MMG5_ARG_pMesh: initialization of a mesh inside the parmesh.
   * MMG5_ARG_pMet: init a metric inside the parmesh
   * PMMG_ARG_dim: next arg will be the mesh dimension
   * 3: mesh dimension
   * PMMG_MPIComm: next arg will be the MPI COmmunicator
   * MPI_COMM_WORLD: MPI communicator
   *
   */
  parmesh = NULL;

  PMMG_Init_parMesh(PMMG_ARG_start,
                    PMMG_ARG_ppParMesh,&parmesh,
                    PMMG_ARG_pMesh,PMMG_ARG_pMet,
                    PMMG_ARG_dim,3,PMMG_ARG_MPIComm,MPI_COMM_WORLD,
                    PMMG_ARG_end);

  /** 2) Build a global mesh in MMG5 format */ 
  /** a) give the vertices (12 vertices with 3 coor = array of size 36) */
  double vert_coor_all[36] = { 0.0, 0.0, 0.0,
                               0.5, 0.0, 0.0,
                               0.5, 0.0, 1.0,
                               0.0, 0.0, 1.0,
                               0.0, 1.0, 0.0,
                               0.5, 1.0, 0.0,
                               0.5, 1.0, 1.0,
                               0.0, 1.0, 1.0,
                               1.0, 0.0, 0.0,
                               1.0, 1.0, 0.0,
                               1.0, 0.0, 1.0,
                               1.0, 1.0, 1.0  };

  int  vert_ref_all[12] = {0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  };

  /** b) give the tetrahedras (12 tetra with 4 vertices = array of size 48) */
  int tetra_vert_all[48] = { 1,  4,  2,  8,
                             8,  3,  2,  7,
                             5,  2,  6,  8,
                             5,  8,  1,  2,
                             7,  2,  8,  6,
                             2,  4,  3,  8,
                             9,  2,  3,  7,
                             7, 11,  9, 12,
                             6,  9, 10,  7,
                             6,  7,  2,  9,
                             12, 9,  7, 10,
                             9,  3, 11,  7  };

  int tetra_ref_all[12] = {1  ,1  ,1  ,1  ,1  ,1  ,2  ,2  ,2  ,2  ,2  ,2  };

  /** c) give the triangles (20 tria with 3 vertices = array of size 60  */
  int tria_vert_all[66] = { 1,  4,  8,
                            1,  2,  4,
                            8,  3,  7,
                            5,  8,  6,
                            5,  6,  2,
                            5,  2,  1,
                            5,  1,  8,
                            7,  6,  8,
                            4,  3,  8,
                            2,  3,  4,
                            9,  3,  2,
                            11, 9, 12,
                            7, 11, 12,
                            6,  7, 10,
                            6, 10,  9,
                            6,  9,  2,
                            12,10,  7,
                            12, 9, 10,
                            3, 11,  7,
                            9, 11,  3,
                            7,  6,  2,
                            7,  3,  2 };

  int tria_ref_all[22] = { 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                           4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                           3, 3 };

  /** d) give solutions values and positions */
  double met_all[12] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};


  /** 3) Manually partition the global mesh */
  /** b) Create local meshes */
  int nv, nt, ne, ncomm, *color_out, *ntifc, *npifc;
  switch( parmesh->nprocs ) {
    case 2:
      /* partitioning into 4 procs */
      nv = 8;
      nt = 12;
      ne = 6;
      ncomm = 1;
      color_out = (int *) malloc(ncomm*sizeof(int));
      ntifc = (int *) malloc(ncomm*sizeof(int));
      npifc = (int *) malloc(ncomm*sizeof(int));
      color_out[0] = (parmesh->myrank+1)%2;
      ntifc[0] = 2;
      npifc[0] = 4;
      break;
    case 4 :
      /* partitioning into 4 procs */
      nv = 6;
      nt = 8;
      ne = 3;
      switch( rank ) {
        case 0 :
          ncomm = 2;
          color_out = (int *) malloc(ncomm*sizeof(int));
          ntifc = (int *) malloc(ncomm*sizeof(int));
          npifc = (int *) malloc(ncomm*sizeof(int));
          color_out[0] = 1;
          ntifc[0] = 1;
          npifc[0] = 3;
          color_out[1] = 3;
          ntifc[1] = 1;
          npifc[1] = 3;
          break;
        case 1 :
          ncomm = 3;
          color_out = (int *) malloc(ncomm*sizeof(int));
          ntifc = (int *) malloc(ncomm*sizeof(int));
          npifc = (int *) malloc(ncomm*sizeof(int));
          color_out[0] = 0;
          ntifc[0] = 1;
          npifc[0] = 3;
          color_out[1] = 2;
          ntifc[1] = 1;
          npifc[1] = 3;
          color_out[2] = 3;
          ntifc[2] = 2;
          npifc[2] = 4;
          break;
        case 2 :
          ncomm = 2;
          color_out = (int *) malloc(ncomm*sizeof(int));
          ntifc = (int *) malloc(ncomm*sizeof(int));
          npifc = (int *) malloc(ncomm*sizeof(int));
          color_out[0] = 1;
          ntifc[0] = 1;
          npifc[0] = 3;
          color_out[1] = 3;
          ntifc[1] = 1;
          npifc[1] = 3;
          break;
        case 3 :
          ncomm = 3;
          color_out = (int *) malloc(ncomm*sizeof(int));
          ntifc = (int *) malloc(ncomm*sizeof(int));
          npifc = (int *) malloc(ncomm*sizeof(int));
          color_out[0] = 0;
          ntifc[0] = 1;
          npifc[0] = 3;
          color_out[1] = 1;
          ntifc[1] = 2;
          npifc[1] = 4;
          color_out[2] = 2;
          ntifc[2] = 1;
          npifc[2] = 3;
          break;
      }
      break;
  }
  double vert_coor[3*nv],met[nv];
  int vert_ref[nv],tetra_vert[4*ne],tetra_ref[ne],tria_vert[3*nt],tria_ref[nt];
  int vert_mask[nv],inv_vert_mask[12],tetra_mask[ne],tria_mask[nt],inv_tria_mask[22];
  int *ifc_tria_loc[ncomm],*ifc_nodes_loc[ncomm];
  int *ifc_tria_glob[ncomm],*ifc_nodes_glob[ncomm];
  /** a) Set interface entities (starting from 1) */

  for( icomm=0; icomm<ncomm; icomm++ ) {
    ifc_tria_loc[icomm] = (int *) malloc(ntifc[icomm]*sizeof(int));
    ifc_tria_glob[icomm] = (int *) malloc(ntifc[icomm]*sizeof(int));
    ifc_nodes_loc[icomm] = (int *) malloc(npifc[icomm]*sizeof(int));
    ifc_nodes_glob[icomm] = (int *) malloc(npifc[icomm]*sizeof(int));
    switch( parmesh->nprocs ) {
      case 2 :
        ifc_tria_glob[0][0] = 21;
        ifc_tria_glob[0][1] = 22;
        ifc_nodes_glob[0][0] = 2;
        ifc_nodes_glob[0][1] = 3;
        ifc_nodes_glob[0][2] = 6;
        ifc_nodes_glob[0][3] = 7;
        break;
    }
  }


  switch( parmesh->myrank ) {
    case 0:
      {
        int vert_mask[8]  = {  0,  1,  2,  3,  4,  5,  6,  7};
        int tetra_mask[6] = {  0,  1,  2,  3,  4,  5};
        int tria_mask[12] = {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 20, 21};
        get_local_mesh(nv, ne, nt, vert_mask, inv_vert_mask, tetra_mask,
                       tria_mask, inv_tria_mask,
                       vert_coor,vert_coor_all,vert_ref,vert_ref_all,
                       tetra_vert,tetra_vert_all,tetra_ref,tetra_ref_all,
                       tria_vert,tria_vert_all,tria_ref,tria_ref_all,
                       met,met_all,ncomm,
                       ntifc,ifc_tria_loc,ifc_tria_glob,
                       npifc,ifc_nodes_loc,ifc_nodes_glob);
        break;
      }
    case 1:
      {
        int vert_mask[8]  = {  1,  2,  5,  6,  8,  9, 10, 11};
        int tetra_mask[6] = {  6,  7,  8,  9, 10, 11};
        int tria_mask[12] = { 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
        get_local_mesh(nv, ne, nt, vert_mask, inv_vert_mask, tetra_mask,
                       tria_mask, inv_tria_mask,
                       vert_coor,vert_coor_all,vert_ref,vert_ref_all,
                       tetra_vert,tetra_vert_all,tetra_ref,tetra_ref_all,
                       tria_vert,tria_vert_all,tria_ref,tria_ref_all,
                       met,met_all,ncomm,
                       ntifc,ifc_tria_loc,ifc_tria_glob,
                       npifc,ifc_nodes_loc,ifc_nodes_glob);
        break;
      }
  }

  /** 4) Pass the mesh in MMG5 format */
  /** Two solutions: just use the PMMG_loadMesh_centralized function that will
      read a .mesh(b) file formatted or manually set your mesh using the
      PMMG_Set* functions */

  /** with ParMmg setters */
  /** a) give the size of the mesh */
  int nVertices       = nv;
  int nTetrahedra     = ne;
  int nPrisms         = 0;
  int nTriangles      = nt;
  int nQuadrilaterals = 0;
  int nEdges          = 0;

  if ( PMMG_Set_meshSize(parmesh,nVertices,nTetrahedra,nPrisms,nTriangles,
                            nQuadrilaterals,nEdges) != 1 ) {
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }

  /** b) set vertices, tetrahedra, triangles */
  if ( !opt ) {
    /* By array: give the array of the vertices coordinates such as the
     * coordinates of the k^th point are stored in vert_coor[3*(k-1)],
     * vert_coor[3*(k-1)+1] and vert_coor[3*(k-1)+2] */
    if ( PMMG_Set_vertices(parmesh,vert_coor,vert_ref) != 1 ) {
      MPI_Finalize();
      exit(EXIT_FAILURE);
    }
  }
  else {
    /* Vertex by vertex: for each vertex, give the coordinates, the reference
       and the position in mesh of the vertex */
    for ( k=0; k<nVertices; ++k ) {
      pos = 3*k;
      if ( PMMG_Set_vertex(parmesh,vert_coor[pos],vert_coor[pos+1],vert_coor[pos+2],
                           vert_ref[k], k+1) != 1 ) {
        MPI_Finalize();
        exit(EXIT_FAILURE);
      }
    }
  }


  if ( !opt ) {
    /* By array: give the array of the tetra vertices and the array of the tetra
     * references. The array of the tetra vertices is such as the four
     * vertices of the k^th tetra are respectively stored in
     * tetra_vert[4*(k-1)],tetra_vert[4*(k-1)+1],tetra_vert[4*(k-1)+2] and
     * tetra_vert[4*(k-1)+3]. */
    if ( PMMG_Set_tetrahedra(parmesh,tetra_vert,tetra_ref) != 1 ) {
      MPI_Finalize();
      exit(EXIT_FAILURE);
    }
  }
  else {
    /* Vertex by vertex: for each tetrahedra,
      give the vertices index, the reference and the position of the tetra */
    for ( k=0; k<nTetrahedra; ++k ) {
      pos = 4*k;

      if ( PMMG_Set_tetrahedron(parmesh,tetra_vert[pos],tetra_vert[pos+1],
                                tetra_vert[pos+2],tetra_vert[pos+3],tetra_ref[k],k+1) != 1 ) {
        MPI_Finalize();
        exit(EXIT_FAILURE);
      }
    }
  }


  if ( !opt ) {
    /* By array: give the array of the tria vertices and the array of the tria
     * references. The array of the tria vertices is such as the three
     * vertices of the k^th tria are stored in
     * tria_vert[3*(k-1)], tria_vert[3*(k-1)+1] and tria_vert[4*(k-1)+2] */
    if ( PMMG_Set_triangles(parmesh,tria_vert,tria_ref) != 1 ) {
      MPI_Finalize();
      exit(EXIT_FAILURE);
    }
  }
  else {
    /* Vertex by vertex: for each triangle, give the vertices index, the
     * reference and the position of the triangle */
    for ( k=0; k<nTriangles; ++k ) {
      pos = 3*k;
      if ( PMMG_Set_triangle(parmesh,
                             tria_vert[pos],tria_vert[pos+1],tria_vert[pos+2],
                             tria_ref[k],k+1) != 1 ) {
        MPI_Finalize();
        exit(EXIT_FAILURE);
      }
    }
  }


  /** 5) Build metric in ParMmg format */
  /** Two solutions: just use the PMMG_loadMet_centralized function that will
      read a .sol(b) file formatted or manually set your sol using the
      PMMG_Set* functions */

  /** Manually set of the metric */
  /** a) give info for the metric structure: metric applied on vertex entities,
      number of vertices, the metric is scalar*/
  if ( PMMG_Set_metSize(parmesh,MMG5_Vertex,nVertices,MMG5_Scalar) != 1 ) {
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }


  if ( !opt ) {
    /* by array */
    if ( PMMG_Set_scalarMets(parmesh,met) != 1 ) {
      MPI_Finalize();
      exit(EXIT_FAILURE);
    }
  }
  else {
    /* vertex by vertex */
    for ( k=0; k<nVertices ; k++ ) {
      if ( PMMG_Set_scalarMet(parmesh,met[k],k+1) != 1 ) {
        MPI_Finalize();
        exit(EXIT_FAILURE);
      }
    }
  }


  /** 6) Build solutions in PMMG format */
  /** Two solutions: just use the PMMG_loadAllSols_centralized function that
      will read a .sol(b) file formatted or manually set your solutions using
      the PMMG_Set* functions */

  /** With parmmg setters: 3 sols per vertex, the first is scalar,
      the second vectorial, the third tensorial  */
  const int nSolsAtVertices = 3; // 3 solutions per vertex
  int solType[3] = {MMG5_Scalar,MMG5_Vector,MMG5_Tensor};

  if ( PMMG_Set_solsAtVerticesSize(parmesh,nSolsAtVertices,nVertices,solType) != 1 ) {
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }


  /** a) give solutions values and positions:
   - First solution (scalar) is equal to x^2 + y^2 + z^2
   - Second (vector) is (x,y,z)
   - Third (Tensor) is (100,0,0,100/(z+1),0,100/(z*z+1))
  */
  double scalar_sol[12],vector_sol[36],tensor_sol[72];
  for ( k=0; k<nVertices; k++ ) {
    pos = 3*k;

    /* First solution */
    scalar_sol[k] = vert_coor[pos]*vert_coor[pos]
      + vert_coor[pos+1]*vert_coor[pos+1]
      + vert_coor[pos+2]*vert_coor[pos+2];

    /* Second */
    vector_sol[3*k]   = vert_coor[pos];
    vector_sol[3*k+1] = vert_coor[pos+1];
    vector_sol[3*k+2] = vert_coor[pos+2];

    /* Third */
    tensor_sol[6*k]   = 100.;
    tensor_sol[6*k+1] = 0.;
    tensor_sol[6*k+2] = 0.;
    tensor_sol[6*k+3] = 100./(vert_coor[pos+2]+1.);
    tensor_sol[6*k+4] = 0.;
    tensor_sol[6*k+5] = 100./(vert_coor[pos+2]*vert_coor[pos+2]+1.);
  }

  if ( !opt ) {
    /* Give the solution by array */
    /* First solution */
    if ( PMMG_Set_ithSols_inSolsAtVertices(parmesh,1,scalar_sol) != 1 ) {
      MPI_Finalize();
      exit(EXIT_FAILURE);
    }
    /* Second */
    if ( PMMG_Set_ithSols_inSolsAtVertices(parmesh,2,vector_sol) != 1 ) {
      MPI_Finalize();
      exit(EXIT_FAILURE);
    }
    /* Third */
    if ( PMMG_Set_ithSols_inSolsAtVertices(parmesh,3,tensor_sol) != 1 ) {
      MPI_Finalize();
      exit(EXIT_FAILURE);
    }
  }
  else {
    /* Vertex by vertex */
    for ( k=0; k<nVertices; k++ ) {
      /* First solution */
      if ( PMMG_Set_ithSol_inSolsAtVertices(parmesh,1,&(scalar_sol[k]),k+1) != 1 ) {
        MPI_Finalize();
        exit(EXIT_FAILURE);
      }
      /* Second */
      pos = 3*k;
      if ( PMMG_Set_ithSol_inSolsAtVertices(parmesh,2,&(vector_sol[pos]),k+1) != 1 ) {
        MPI_Finalize();
        exit(EXIT_FAILURE);
      }
      /* Third */
      pos = 6*(k-1);
      if ( PMMG_Set_ithSol_inSolsAtVertices(parmesh,3,&(tensor_sol[pos]),k+1) != 1 ) {
        MPI_Finalize();
        exit(EXIT_FAILURE);
      }
    }
  }

  
  /** ------------------------------ STEP  II -------------------------- */
  /** initialization of interface communicators */
 
  switch( API_mode ) {
    
    case PMMG_APIDISTRIB_faces :
      if( !rank ) printf("\n--- API mode: Setting face communicators\n");
      
      ier = PMMG_Set_numberOfFaceCommunicators(parmesh, ncomm);
      for( icomm=0; icomm<ncomm; icomm++ ) {
        ier = PMMG_Set_ithFaceCommunicatorSize(parmesh, icomm,
                                               color_out[icomm],
                                               ntifc[icomm]);
        ier = PMMG_Set_ithFaceCommunicator_faces(parmesh, icomm,
                                                 ifc_tria_loc[icomm],
                                                 ifc_tria_glob[icomm], 1 );
      }
      break;
    
    case PMMG_APIDISTRIB_nodes :
      if( !rank ) printf("\n--- API mode: Setting node communicators\n");
      
      ier = PMMG_Set_numberOfNodeCommunicators(parmesh, ncomm);
      for( icomm=0; icomm<ncomm; icomm++ ) {
        ier = PMMG_Set_ithNodeCommunicatorSize(parmesh, icomm,
                                               color_out[icomm],
                                               npifc[icomm]);
        ier = PMMG_Set_ithNodeCommunicator_nodes(parmesh, icomm,
                                                 ifc_nodes_loc[icomm],
                                                 ifc_nodes_glob[icomm], 1 );
      }
      break;
  }
 
  /** ------------------------------ STEP III -------------------------- */
  /** remesh function */
  ierlib = PMMG_parmmglib_distributed( parmesh );

  if ( ierlib != PMMG_STRONGFAILURE ) {

    /** ------------------------------ STEP  IV -------------------------- */
    /** recover parallel interfaces */
  
    int next_node_comm,next_face_comm,nitem_node_comm,nitem_face_comm;
    ier = PMMG_Get_numberOfNodeCommunicators(parmesh,&next_node_comm);
    ier = PMMG_Get_numberOfFaceCommunicators(parmesh,&next_face_comm);
  
    free(color_out);
    color_out = (int *) malloc(next_node_comm*sizeof(int));
    for( icomm=0; icomm<next_node_comm; icomm++ )
      ier = PMMG_Get_ithNodeCommunicatorSize(parmesh, icomm, &color_out[icomm],
                                             &nitem_node_comm);
 
    free(color_out);
    color_out = (int *) malloc(next_face_comm*sizeof(int));
    for( icomm=0; icomm<next_face_comm; icomm++ )
      ier = PMMG_Get_ithFaceCommunicatorSize(parmesh, icomm, &color_out[icomm],
                                             &nitem_face_comm);
  
    int **out_tria_loc, **out_node_loc;
   
    out_node_loc = (int **) malloc(next_node_comm*sizeof(int *));
    out_node_loc[0] = (int *) malloc(nitem_node_comm*sizeof(int));
    ier = PMMG_Get_NodeCommunicator_nodes(parmesh, out_node_loc);
  
    for( ier=0; ier < nitem_node_comm; ier++ )
      printf("rank %d node %d\n",parmesh->myrank,out_node_loc[0][ier]);
  
    out_tria_loc = (int **) malloc(next_face_comm*sizeof(int *));
    out_tria_loc[0] = (int *) malloc(nitem_face_comm*sizeof(int));
    ier = PMMG_Get_FaceCommunicator_faces(parmesh, out_tria_loc);
  
    for( ier=0; ier < nitem_face_comm; ier++ )
      printf("rank %d tria %d\n",parmesh->myrank,out_tria_loc[0][ier]);


    /** ------------------------------ STEP III -------------------------- */
    /** get results */
    /** Two solutions: just use the PMMG_saveMesh/PMMG_saveSol functions
        that will write .mesh(b)/.sol formatted files or manually get your mesh/sol
        using the PMMG_getMesh/PMMG_getSol functions */

    /** 1) Get the mesh with ParMmg getters and save it at the Medit file format */
    if( !(inm = fopen(fileout,"w")) ) {
      fprintf(stderr,"  ** UNABLE TO OPEN OUTPUT MESH FILE.\n");
      exit(EXIT_FAILURE);
    }
    fprintf(inm,"MeshVersionFormatted 2\n");
    fprintf(inm,"\nDimension 3\n");

    /** a) get the size of the mesh: vertices, tetra, triangles, edges and
     * allocate the arrays to receive data */
    nVertices   = 0;
    nTetrahedra = 0;
    nTriangles  = 0;
    nEdges      = 0;
    if ( PMMG_Get_meshSize(parmesh,&nVertices,&nTetrahedra,NULL,&nTriangles,NULL,
                           &nEdges) !=1 ) {
      ier = PMMG_STRONGFAILURE;
    }

    /* Table to store the vertices */
    double *vert = (double*)calloc((nVertices)*3,sizeof(double));
    if ( !vert ) {
      perror("  ## Memory problem: point calloc");
      nVertices = 0;
      ier = PMMG_STRONGFAILURE;
    }

    /* Table to store the tetra */
    int *tetra = (int*)calloc((nTetrahedra)*4,sizeof(int));
    if ( !tetra ) {
      perror("  ## Memory problem: tetra calloc");
      nTetrahedra = 0;
      ier = PMMG_STRONGFAILURE;
    }

    /* Table to store the tria */
    int *tria = (int*)calloc((nTriangles)*3,sizeof(int));
    if ( !tria ) {
      perror("  ## Memory problem: tria calloc");
      nTriangles = 0;
      ier = PMMG_STRONGFAILURE;
    }

    /* Table to store the edges */
    int *edge = (int*)calloc((nEdges)*2,sizeof(int));
    if ( !edge ) {
      perror("  ## Memory problem: edge calloc");
      nEdges = 0;
      ier = PMMG_STRONGFAILURE;
    }

    /* Table to store the vertices/tetra/triangles/edges references */
    int *ref = (int*)calloc(MAX4(nVertices,nTetrahedra,nTriangles,nEdges),sizeof(int));
    if ( !ref ) {
      perror("  ## Memory problem: ref calloc");
      MPI_Finalize();
      exit(EXIT_FAILURE);
    }

    /* Table to know if a vertex is corner */
    int *corner = (int*)calloc(nVertices,sizeof(int));
    if ( !corner ) {
      perror("  ## Memory problem: corner calloc");
      MPI_Finalize();
      exit(EXIT_FAILURE);
    }

    /* Table to know if a vertex/tetra/tria/edge is required */
    int *required = (int*)calloc(MAX4(nVertices,nTetrahedra,nTriangles,nEdges),sizeof(int));
    if ( !required ) {
      perror("  ## Memory problem: required calloc");
      MPI_Finalize();
      exit(EXIT_FAILURE);
    }

    /* Table to know if an edge delimits a sharp angle */
    int *ridge = (int*)calloc(nEdges ,sizeof(int));
    if ( !ridge ) {
      perror("  ## Memory problem: ridge calloc");
      MPI_Finalize();
      exit(EXIT_FAILURE);
    }

    /** b) Vertex recovering */
    nreq = nc = 0;
    fprintf(inm,"\nVertices\n%d\n",nVertices);

    if ( !opt ) {
      /* By array */
      if ( PMMG_Get_vertices(parmesh,vert,ref,corner,required) != 1 ) {
        fprintf(inm,"Unable to get mesh vertices \n");
        ier = PMMG_STRONGFAILURE;
      }
      for ( k=0; k<nVertices; k++ ) {
        if ( corner && corner[k] )  nc++;
        if ( required && required[k] )  nreq++;
      }
    }
    else {
      /* Vertex by vertex */
      for ( k=0; k<nVertices; k++ ) {
        pos = 3*k;
        if ( PMMG_Get_vertex(parmesh,&(vert[pos]),&(vert[pos+1]),&(vert[pos+2]),
                             &(ref[k]),&(corner[k]),&(required[k])) != 1 ) {
          fprintf(inm,"Unable to get mesh vertex %d \n",k);
          ier = PMMG_STRONGFAILURE;
        }
        if ( corner && corner[k] )  nc++;
        if ( required && required[k] )  nreq++;
      }
    }
    for ( k=0; k<nVertices; k++ ) {
      pos = 3*k;
      fprintf(inm,"%.15lg %.15lg %.15lg %d \n",vert[pos],vert[pos+1],vert[pos+2],ref[k]);
    }

    fprintf(inm,"\nCorners\n%d\n",nc);
    for ( k=0; k<nVertices; k++ ) {
      if ( corner && corner[k] )  fprintf(inm,"%d \n",k);
    }
    fprintf(inm,"\nRequiredVertices\n%d\n",nreq);
    for ( k=0; k<nVertices; k++ ) {
      if ( required && required[k] )  fprintf(inm,"%d \n",k);
    }
    free(corner);
    corner = NULL;

    /** d) Triangles recovering */
    nreq = 0;
    fprintf(inm,"\nTriangles\n%d\n",nTriangles);

    if ( !opt ) {
      /* By array */
      if ( PMMG_Get_triangles(parmesh,tria,ref,required) != 1 ) {
        fprintf(inm,"Unable to get mesh triangles\n");
        ier = PMMG_STRONGFAILURE;
      }
      for ( k=0; k<nTriangles; k++ ) {
        if ( required && required[k] )  nreq++;
      }
    }
    else {
      /* Triangle by triangle */
      for ( k=0; k<nTriangles; k++ ) {
        pos = 3*k;
        if ( PMMG_Get_triangle(parmesh,&(tria[pos]),&(tria[pos+1]),&(tria[pos+2]),
                               &(ref[k]),&(required[k])) != 1 ) {
          fprintf(inm,"Unable to get mesh triangle %d \n",k);
          ier = PMMG_STRONGFAILURE;
        }
        if ( required && required[k] )  nreq++;
      }
    }
    for ( k=0; k<nTriangles; k++ ) {
      pos = 3*k;
      fprintf(inm,"%d %d %d %d \n",tria[pos],tria[pos+1],tria[pos+2],ref[k]);
    }


    fprintf(inm,"\nRequiredTriangles\n%d\n",nreq);
    for ( k=0; k<nTriangles; k++ ) {
      if ( required && required[k] )  fprintf(inm,"%d \n",k);
    }

    /** e) Edges recovering */
    nreq = 0;nr = 0;
    fprintf(inm,"\nEdges\n%d\n",nEdges);

    if ( !opt ) {
      /* By array */
      if ( PMMG_Get_edges(parmesh,edge,ref,ridge,required) != 1 ) {
        fprintf(inm,"Unable to get mesh edges\n");
        ier = PMMG_STRONGFAILURE;
      }
      for ( k=0; k<nEdges; k++ ) {
        if ( ridge && ridge[k] )  nr++;
        if ( required && required[k] )  nreq++;
      }
    }
    else {
      /* Edge by edge */
      for ( k=0; k<nEdges; k++ ) {
        pos = 2*k;
        if ( PMMG_Get_edge(parmesh,&(edge[pos]),&(edge[pos+1]),
                           &(ref[k]),&(ridge[k]),&(required[k])) != 1 ) {
          fprintf(inm,"Unable to get mesh edge %d \n",k);
          ier = PMMG_STRONGFAILURE;
        }
        if ( ridge && ridge[k] )  nr++;
        if ( required && required[k] )  nreq++;
      }
    }
    for ( k=0; k<nEdges; k++ ) {
      pos = 2*k;
      fprintf(inm,"%d %d %d \n",edge[pos],edge[pos+1],ref[k]);
    }

    fprintf(inm,"\nRequiredEdges\n%d\n",nreq);
    for ( k=0; k<nEdges; k++ ) {
      if ( required && required[k] )  fprintf(inm,"%d \n",k);
    }
    fprintf(inm,"\nRidges\n%d\n",nr);
    for ( k=0; k<nEdges; k++ ) {
      if ( ridge && ridge[k] )  fprintf(inm,"%d \n",k);
    }

    /** c) Tetra recovering */
    nreq = 0;
    fprintf(inm,"\nTetrahedra\n%d\n",nTetrahedra);

    if ( !opt ) {
      /* By array */
      if ( PMMG_Get_tetrahedra(parmesh,tetra,ref,required) != 1 ) {
        fprintf(inm,"Unable to get mesh tetra\n");
        ier = PMMG_STRONGFAILURE;
      }
      for ( k=0; k<nTetrahedra; k++ ) {
        if ( required && required[k] )  nreq++;
      }
    }
    else {
      /* Tetra by tetra */
      for ( k=0; k<nTetrahedra; k++ ) {
        pos = 4*k;
        if ( PMMG_Get_tetrahedron(parmesh,
                                  &(tetra[pos  ]),&(tetra[pos+1]),
                                  &(tetra[pos+2]),&(tetra[pos+3]),
                                  &(ref[k]),&(required[k])) != 1 ) {
          fprintf(inm,"Unable to get mesh tetra %d \n",k);
          ier = PMMG_STRONGFAILURE;
        }
        if ( required && required[k] )  nreq++;
      }
    }
    for ( k=0; k<nTetrahedra; k++ ) {
      pos = 4*k;
      fprintf(inm,"%d %d %d %d %d \n",
              tetra[pos],tetra[pos+1],tetra[pos+2],tetra[pos+3],ref[k]);
    }

    fprintf(inm,"\nRequiredTetrahedra\n%d\n",nreq);
    for ( k=0; k<nTetrahedra; k++ ) {
      if ( required && required[k] )  fprintf(inm,"%d \n",k);
    }

    fprintf(inm,"\nEnd\n");
    fclose(inm);

    free(tetra)   ; tetra    = NULL;
    free(ref)     ; ref      = NULL;
    free(required); required = NULL;
    free(ridge)   ; ridge    = NULL;

    /** 3) Get the metric with ParMmg getters */
    if ( !(inm = fopen(metout,"w")) ) {
      fprintf(stderr,"  ** UNABLE TO OPEN OUTPUT METRIC FILE.\n");
      exit(EXIT_FAILURE);
    }
    fprintf(inm,"MeshVersionFormatted 2\n");
    fprintf(inm,"\nDimension 3\n");

    /** a) get the size of the metric: type of entity to which apply the
        metric(SolAtVertices,...), number of entities to which apply the metric,
        type of solution (scalar, tensor...) */
    nVertices = 0;
    int typEntity,typSol;
    if ( PMMG_Get_metSize(parmesh,&typEntity,&nVertices,&typSol) != 1 ) {
      printf("Unagle to get metric size\n");
      nVertices = 0;
      ier = PMMG_LOWFAILURE;
    }

    /* We set a scalar metric so the output metric must be scalar */
    if ( ( typEntity != MMG5_Vertex )  || ( typSol != MMG5_Scalar ) ) {
      MPI_Finalize();
      exit(EXIT_FAILURE);
    }

    /** b) Vertex recovering */
    double *sol = (double*)calloc(nVertices+1 ,sizeof(double));

    fprintf(inm,"\nSolAtVertices\n%d\n",nVertices);
    fprintf(inm,"1 1 \n\n");
    if ( !opt ) {
      /* by array */
      if ( PMMG_Get_scalarMets(parmesh,sol) != 1 ) {
        fprintf(inm,"Unable to get metrics\n");
        ier = PMMG_LOWFAILURE;
      }
    }
    else {
      for ( k=0; k<nVertices; k++ ) {
        /* Vertex by vertex */
        if ( PMMG_Get_scalarMet(parmesh,&sol[k]) != 1 ) {
          fprintf(inm,"Unable to get metrics %d \n",k);
          ier = PMMG_LOWFAILURE;
        }
      }
    }
    for ( k=0; k<nVertices; k++ ) {
      fprintf(inm,"%.15lg \n",sol[k]);
    }

    fprintf(inm,"\nEnd\n");
    fclose(inm);

    /** 4) Get the solutions with ParMmg getters */
    // To implement when ParMmg will be ready
  }
  else if ( ierlib == PMMG_STRONGFAILURE ) {
    fprintf(stdout,"BAD ENDING OF PARMMGLIB: UNABLE TO SAVE MESH\n");
  }

  /** 5) Free the PMMG5 structures */
  PMMG_Free_all(PMMG_ARG_start,
                PMMG_ARG_ppParMesh,&parmesh,
                PMMG_ARG_end);

  free(fileout); fileout = NULL;
  free(solout) ; solout  = NULL;
  free(metout) ; metout  = NULL;

  MPI_Finalize();

  return ierlib;
}
