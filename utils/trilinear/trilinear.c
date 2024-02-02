/*

Author: Pablo Benitez-Llambay
Email: pbllambay@gmail.com
Date: 6/6/2017

Purpose: Trilinear interpolation from a spherical
         mesh into a cube

*/


//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/ndarraytypes.h>
#include <numpy/arrayobject.h>

typedef double real;  //Data is assumed double
typedef double realc; //Cubic mesh

real max(real *v, int size) {
  real a = -1e30;
  int i;
  
  for (i=0; i<size; i++) {
    if (v[i]>a)
      a = v[i];
  }
  return a;
}

real min(real *v, int size) {
  real a = 1e30;
  int i;
  for (i=0; i<size; i++) {
    if (v[i]<a)
      a = v[i];
  }
  return a;
}

realc trilinear_interpolation(real x0, real x1,
			      real y0, real y1,
			      real z0, real z1,
			      real c000, real c100, real c110, real c010,
			      real c001, real c101, real c111, real c011,
			      realc x, realc y, realc z) {
  //The cube is defined by the vertices (x0,y0,z0), (x1,y1,z1)
  real xd,yd,zd;
  real c00, c01, c10, c11;
  real c0, c1;

  xd = (x-x0)/(x1-x0);
  yd = (y-y0)/(y1-y0);
  zd = (z-z0)/(z1-z0);
  // Linear interpolation along x
  c00 = c000*(1.0-xd) + c100*xd;
  c01 = c001*(1.0-xd) + c101*xd;
  c10 = c010*(1.0-xd) + c110*xd;
  c11 = c011*(1.0-xd) + c111*xd;
  // Now linear interpolation along y
  c0  = c00*(1.0-yd) + c10*yd;
  c1  = c01*(1.0-yd) + c11*yd;
  // Finally, linear interpolation along z
  return (realc)(c0*(1-zd) + c1*zd);
    
}

realc *_trilinear(real *field,
		  real *r, real* phi, real *theta,
		  int nx, int ny, int nz,
		  realc *xc, realc *yc, realc *zc,
		  int nxc, int nyc, int nzc, realc minval) {
  
  //data is staggered or not depending on phi, r, theta values
  
  realc  rc, phic, thetac;
  int i,j,k;
  int ix, iy, iz;

  real theta_max = max(theta,nz);
  real theta_min = min(theta,nz);
  real r_max     = max(r,ny);
  real r_min     = min(r,ny);
  real phi_min   = min(phi,nx);
  real phi_max   = max(phi,nx);

  realc *output = (realc*)malloc(sizeof(realc)*nxc*nyc*nzc);
  
  for (k=0; k<nzc; k++) {
    for (j=0; j<nyc; j++) {
      for (i=0; i<nxc; i++) {

	//We fisrt determine the spherical coordinates of the cubic mesh.
	thetac = atan2(sqrt(xc[i]*xc[i]+yc[j]*yc[j]),zc[k]);
	rc     = sqrt(xc[i]*xc[i]+yc[j]*yc[j]+zc[k]*zc[k]);
	phic   = atan2(yc[j],xc[i]);

	iz = (int)(((thetac-theta_min)/(theta_max-theta_min)*(real)(nz-1)));
	iy = (int)(((rc-r_min)/(r_max-r_min)*(real)(ny-1)));
	ix = (int)(((phic-phi_min)/(phi_max-phi_min)*(real)(nx-1)));
	
	if (iz>=0 && iz<nz-1 && iy>=0 && iy<ny-1 && ix>=0 && ix<nx-1)  {
	  output[i+j*nxc+k*nxc*nyc] = trilinear_interpolation(phi[ix], phi[ix+1],
							      r[iy], r[iy+1],
							      theta[iz], theta[iz+1],
							      field[ix+iy*(nx)+iz*(nx)*ny],
							      field[(ix+1)+iy*(nx)+iz*(nx)*ny],
							      field[(ix+1)+(iy+1)*(nx)+iz*(nx)*ny],
							      field[(ix)+(iy+1)*(nx)+iz*(nx)*ny],
							      field[(ix)+(iy)*(nx)+(iz+1)*(nx)*ny],
							      field[(ix+1)+(iy)*(nx)+(iz+1)*(nx)*ny],
							      field[(ix+1)+(iy+1)*(nx)+(iz+1)*(nx)*ny],
							      field[(ix)+(iy+1)*(nx)+(iz+1)*(nx)*ny],
							      phic, rc, thetac);
	}
	else {
	  output[i+j*nxc+k*nxc*nyc] = minval;
	}
      }
    }
  }
  return output;
}

static PyObject *trilinear(PyObject* self, PyObject* args) {

  PyArrayObject *field, *phi, *r, *theta;
  PyArrayObject *xc, *yc, *zc;
  real minval;

  import_array();
    
  if (!PyArg_ParseTuple(args, "OOOOOOO|d", &field,&r,&phi,&theta,&xc,&yc,&zc,&minval))
    return NULL;

  int nz = (int)field->dimensions[0];
  int ny = (int)field->dimensions[1];
  int nx = (int)field->dimensions[2];

  int nxc = (int)xc->dimensions[0];
  int nyc = (int)yc->dimensions[0];
  int nzc = (int)zc->dimensions[0];

  printf("Processing cube...\n");
  real* output_cube = _trilinear((real*)field->data,
				 (real*)r->data, (real*)phi->data, (real*)theta->data,
				 nx, ny, nz,
				 (realc*)xc->data, (realc*)yc->data, (realc*)zc->data,
				 nxc, nyc, nzc, minval);
  printf("Cube ready\n");
  
  npy_intp dims[3] = {nzc,nyc,nxc};
  PyObject *OutputArray = PyArray_SimpleNewFromData(3, dims, NPY_FLOAT64, output_cube);
  
  return Py_BuildValue("O", OutputArray);
}

static PyMethodDef TrilinearMethods[] = {
  {"trilinear", trilinear, METH_VARARGS, "Interpolating a sphere into a cube."},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef trilinearmodule = {
    PyModuleDef_HEAD_INIT,
    "trilinear",  // nombre del módulo
    NULL,  // Documentación del módulo, puede ser NULL
    -1,  // Tamaño del módulo, o -1 si el módulo mantiene estado en variables globales
    TrilinearMethods  // Métodos del módulo
};

// PyMODINIT_FUNC inittrilinear(void) {
//   (void) Py_InitModule("trilinear", TrilinearMethods);
// }

PyMODINIT_FUNC PyInit_trilinear(void) {
    import_array();  // Necesario para inicializar las API de C de NumPy
    return PyModule_Create(&trilinearmodule);
}
