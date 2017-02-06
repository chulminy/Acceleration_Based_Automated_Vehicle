/*=================================================================
* GentleBoost.cpp
*
* (Description)
*
* This is a MEX-file for MATLAB.
* A debugger linker error is frequently happened.
* The only solution that I found is restarting your matlab machine.
* Use following two lines in Matlab  (not to be in a panic)
system(['start matlab.exe -sd ' pwd()]);
system(['taskkill /f /pid ' num2str(feature('getpid'))]);
*=================================================================*/

#include "mex.h"
#include <math.h>		/* floor*/
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <vector>		// std::vector
#include <functional>
#include <iostream>
#include <algorithm>    
#include <numeric>
#include "matrix.h"
#include <omp.h>

#pragma comment(lib, "libmwservices.lib")
extern bool ioFlush(void);

//using namespace std;
extern void _main();

const int numInputArgs = 3;
const int numOutputArgs = 4;

template <typename T>
std::vector<size_t> ordered(std::vector<T> const& values) {
	std::vector<size_t> indices(values.size());
	std::iota(begin(indices), end(indices), static_cast<size_t>(0));

	std::sort(
		begin(indices), end(indices),
		[&](size_t a, size_t b) { return values[a] < values[b]; }
	);
	return indices;
}
// Act like matlab's [Y,I] = SORT(X)
// Input:
//   unsorted  unsorted vector
// Output:
//   sorted     sorted vector, allowed to be same as unsorted
//   index_map  an index map such that sorted[i] = unsorted[index_map[i]]
template <class T>
void sort(
	std::vector<T> &unsorted,
	std::vector<T> &sorted,
	std::vector<size_t> &index_map);
// Act like matlab's Y = X[I]
// where I contains a vector of indices so that after,
// Y[j] = X[I[j]] for index j
// this implies that Y.size() == I.size()
// X and Y are allowed to be the same reference
template< class T >
void reorder(
	std::vector<T> & unordered,
	std::vector<size_t> const & index_map,
	std::vector<T> & ordered);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

// Comparison struct used by sort
// http://bytes.com/topic/c/answers/132045-sort-get-index
template<class T> struct index_cmp
{
	index_cmp(const T arr) : arr(arr) {}
	bool operator()(const size_t a, const size_t b) const
	{
		return arr[a] < arr[b];
	}
	const T arr;
};

template <class T>
void sort(
	std::vector<T> & unsorted,
	std::vector<T> & sorted,
	std::vector<size_t> & index_map)
{
	// Original unsorted index map
	index_map.resize(unsorted.size());
	for (size_t i = 0; i < unsorted.size(); i++)
	{
		index_map[i] = i;
	}
	// Sort the index map, using unsorted for comparison
	sort(
		index_map.begin(),
		index_map.end(),
		index_cmp<std::vector<T>& >(unsorted));

	sorted.resize(unsorted.size());
	reorder(unsorted, index_map, sorted);
}

// This implementation is O(n), but also uses O(n) extra memory
template< class T >
void reorder(
	std::vector<T> & unordered,
	std::vector<size_t> const & index_map,
	std::vector<T> & ordered)
{
	// copy for the reorder according to index_map, because unsorted may also be
	// sorted
	std::vector<T> copy = unordered;
	ordered.resize(index_map.size());
	for (int i = 0; i < index_map.size(); i++)
	{
		ordered[i] = copy[index_map[i]];
	}
}

void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[]) {

    /* Simple "srand()" seed: just use "time()" */
	srand((unsigned)time(0));
	
	//	omp_set_num_threads(4);

	/* Check for proper number of input and output arguments */
	if (nrhs != numInputArgs)
		mexErrMsgTxt("Incorrect number of input arguments");
	if (nlhs != numOutputArgs)
		mexErrMsgTxt("Incorrect number of output arguments");

	/* input arguments */

#define featureVec	 prhs[0]				// featureVec
#define featureIdx	 prhs[1]				// featureIdx

	const int Nrounds = (int)mxGetScalar(prhs[2]);		// marker size

#define k		plhs[0] 			// k
#define th		plhs[1] 			// threshold
#define a		plhs[2] 			// a
#define b		plhs[3] 			// b

	double *pfeatureVec = mxGetPr(featureVec);
	const double *pfeatureIdx = mxGetPr(featureIdx);

	// featureVec size :  Npt_total * Nfeatures;

	const size_t Npt_total = mxGetM(featureVec);
	const size_t Nfeatures = mxGetN(featureVec);

	k = mxCreateDoubleMatrix(1, Nrounds, mxREAL);
	th = mxCreateDoubleMatrix(1, Nrounds, mxREAL);
	a = mxCreateDoubleMatrix(1, Nrounds, mxREAL);
	b = mxCreateDoubleMatrix(1, Nrounds, mxREAL);

	double *pk = mxGetPr(k);
	double *pth = mxGetPr(th);
	double *pa = mxGetPr(a);
	double *pb = mxGetPr(b);

	mxArray *w = mxCreateDoubleMatrix(1, Npt_total, mxREAL);
	double *pw = mxGetPr(w);
	for (int ii = 0; ii < Npt_total; ii++)
		pw[ii] = 1.0;

	std::vector<double> th_vectors(Nfeatures);
	std::vector<double> a_vectors(Nfeatures);
	std::vector<double> b_vectors(Nfeatures);
	std::vector<double> error_vectors(Nfeatures);
	
	std::vector<double> vec_z_org(pfeatureIdx, pfeatureIdx + Npt_total);
	std::vector<double> vec_w_org(Npt_total);

	double sum_w = 0.0;

	size_t kk = 0;

	double fmTmp = 0;

	for (int tt = 0; tt < Nrounds; tt++)
	{
		vec_w_org.assign(pw, pw + Npt_total);

		// threshold will be located in between samples.
		// just in case... (w should be 1)
		sum_w = std::accumulate(vec_w_org.begin(), vec_w_org.end(), 0.0);
		std::transform(vec_w_org.begin(), vec_w_org.end(), vec_w_org.begin(), std::bind2nd
			(std::divides<double>(), sum_w));

#pragma omp parallel for num_threads(4)
		for (int qq = 0; qq < Nfeatures; qq++)
		{
			std::vector<double> vec_x(pfeatureVec + Npt_total*qq, pfeatureVec + Npt_total*(qq + 1));
			std::vector<double> vec_z(Npt_total), vec_w(Npt_total);
			std::vector<double> vec_Szw_tmp(Npt_total), Szw(Npt_total), Sw(Npt_total);
			std::vector<double> aa(Npt_total), bb(Npt_total);
			std::vector<double> Error(Npt_total);

			std::vector<size_t> vec_i(Npt_total);
			
			sort(vec_x, vec_x, vec_i);
			reorder(vec_z_org, vec_i, vec_z);
			reorder(vec_w_org, vec_i, vec_w);
						
			// Szw = cumsum(z.*w)
			std::transform(vec_z.begin(), vec_z.end(), vec_w.begin(), vec_Szw_tmp.begin(), std::multiplies<double>());
			std::partial_sum(vec_Szw_tmp.begin(), vec_Szw_tmp.end(), Szw.begin());

			double Ezw = Szw.back();

			// Szw = cumsum(w)
			std::partial_sum(vec_w.begin(), vec_w.end(), Sw.begin());

			// This is 'a' and 'b' for all posible thresholds :
			std::transform(Szw.begin(), Szw.end(), Sw.begin(), bb.begin(), std::divides<double>());

			double sum_w_z_2 = 0.0;
			for (int ii = 0; ii < Npt_total; ii++)
			{
				if (1 == Sw[ii])
				{
					aa[ii] = (Ezw - Szw[ii]) - bb[ii];
				}
				else
				{
					aa[ii] = (Ezw - Szw[ii]) / (1 - Sw[ii]) - bb[ii];
				}
				sum_w_z_2 = sum_w_z_2 + vec_w[ii] * vec_z[ii] * vec_z[ii];
			}

			// Now, let's look at the error so that we pick the minimum:
			// the error at each threshold is :
			// for i = 1 : Nsamples
			// error(i) = sum(w.*(z - (a(i)*(x>th(i)) + b(i))). ^ 2);
			// end
			// but with vectorized code it is much faster but also more obscure code :
			for (int ii = 0; ii < Npt_total; ii++)
			{
				Error[ii] = sum_w_z_2 - 2 * aa[ii] * (Ezw - Szw[ii]) - 2 * bb[ii] * Ezw;
				Error[ii] = Error[ii] + (aa[ii] * aa[ii] + 2 * aa[ii] * bb[ii])*(1 - Sw[ii]) + bb[ii] * bb[ii];
			}
			sort(Error, Error, vec_i);

			kk = vec_i[0];
			if (kk == Npt_total)
				th_vectors[qq] = vec_x[kk];
			else
				th_vectors[qq] = (vec_x[kk] + vec_x[kk + 1]) / 2.0;

			error_vectors[qq] = Error[0];
			a_vectors[qq] = aa[kk];
			b_vectors[qq] = bb[kk];

		}
		std::vector<size_t> vec_ii(Npt_total);

		sort(error_vectors, error_vectors, vec_ii);

		kk = vec_ii[0];
		pk[tt] = (double)kk;
		pth[tt] = th_vectors[kk];
		pa[tt] = a_vectors[kk];
		pb[tt] = b_vectors[kk];
				
		for (int qq = 0; qq < Npt_total; qq++)
		{
			if (pfeatureVec[(int)pk[tt] * Npt_total + qq] > pth[tt])
				fmTmp = pa[tt] + pb[tt];
			else
				fmTmp = pb[tt];

			pw[qq] = pw[qq] * exp(-1 * pfeatureIdx[qq] * fmTmp);
		}

		if (error_vectors[0] < 0.0001)
		{	
		for (int qq = 0; qq < Npt_total; qq++)
			pfeatureVec[(int)pk[tt] * Npt_total + qq] = (double) (rand()% 1000 +1);
		}
		
		pk[tt] = pk[tt] + 1; // Matlab index correction

	}
	return;
}
