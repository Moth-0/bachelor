#pragma once 

#include<string>
#include<vector>
#include<initializer_list>
#include<functional>
#include<cmath>
#include<cassert>
#include<iostream>
#include<cstdio>
#include<iomanip>
#include<ostream>

// Definitions
#define FORV(i,v) for(size_t i=0;i<v.size();i++)
#define FOR_COLS(i,A) for(size_t i=0;i<A.size2();i++)
#define SELF (*this)

namespace qm{
struct vector {
	std::vector<double> data;
	vector(size_t n) : data(n) {}

	vector(std::initializer_list<double> list) :
		data(list.begin(),list.end()) {}

	vector()			                =default;
	vector(const vector&)		        =default;
	vector(vector&&)		            =default;
    ~vector()                           =default;
	vector& operator=(const vector&)    =default;
	vector& operator=(vector&&)	        =default;

    // For sizing
	size_t size() const {return data.size();}
	void resize(size_t n) {data.resize(n);}

    // For indexing
	double& operator[](size_t i) {return data[i];}
	const double& operator[](size_t i) const {return data[i];}

	vector& operator+=(const vector& other) {
        FORV(i, SELF) data[i]+=other.data[i];
        return SELF;}

	vector& operator-=(const vector& other) {
        FORV(i, SELF) data[i]-=other.data[i];
        return SELF;}

	vector& operator*=(double x) {
        FORV(i, SELF) data[i] *= x;
        return SELF;}

	vector& operator/=(double x) {
        FORV(i, SELF) data[i] /= x; 
        return SELF;}

	vector& add(double x) {
        data.push_back(x);
        return SELF;} 

	vector& push_back(double x) {
        data.push_back(x); 
        return SELF;} 

	double norm() const {
        double s = 0; 
        FORV(i,SELF)s+=SELF[i]*SELF[i];
	    return std::sqrt(s);}

	void print(std::string s="") const {
        std::cout << s;
	    FORV(i,SELF)printf("%9.3g ",(double)SELF[i]);
	    printf("\n");}

	vector map(std::function<double(double)> f) const {
        vector r=SELF;
	    for(size_t i=0;i<r.size();i++)r[i]=f(r[i]);
	    return r;}
};

vector operator+(const vector& v, const vector& u) {
    vector r = v; 
    r += u; 
    return r; }

vector operator-(const vector& v) {
    vector r = v; 
    FORV(i, r) r[i] = -r[i];
    return r;}

vector operator-(const vector& v, const vector& u) {
    vector r = v; 
    r -= u; 
    return r; }

vector operator*(const vector& v, double a) {
    vector r = v; 
    r *= a; 
    return r;} 

vector operator*(double a, const vector& v) {
    vector r = v; 
    r *= a; 
    return r; }

vector operator/(const vector& v, double a) {
    vector r = v; 
    r /= a; 
    return r;}

vector pow(const vector& v, size_t x) {
    vector r = v;
    FORV(i, r) r[i] = std::pow(v[i], x);
    return r;}

long double dot(const vector& v, const vector& u) {
        long double sum = 0;
        FORV(i, v) sum+= v[i] * u[i];
        return sum;
    }

bool approx(double x,double y,double acc=1e-6,double eps=1e-6){
	if(std::fabs(x-y) < acc)return true;
	if(std::fabs(x-y) < eps*(std::fabs(x)+std::fabs(y)))return true;
	return false;}

bool approx(const vector& v, const vector& u, double acc=1e-6, double eps=1e-6){
	if(u.size()!=v.size())return false;
	for(size_t i=0;i<u.size();i++)if(!approx(u[i],v[i],acc,eps))return false;
	return true;}

std::ostream& operator<<(std::ostream& os, const vector& v) {
    os << "(";
    for(size_t i=0;i<v.size()-1;i++) os << v[i] <<", ";
    size_t l = v.size()-1;
    os << v[l] << ")";
    return os;}


// Matrix object 
struct matrix {
	std::vector<vector> cols;
    matrix(std::size_t n,std::size_t m) {
        cols.resize(m);
        for(std::size_t i=0;i<m;i++) cols[i].resize(n);
    }
    
    matrix(std::initializer_list<std::initializer_list<double>> list) {
        for(auto c : list) {
            cols.push_back(vector(c));
        }
    }

	matrix()                    =default;
	matrix(const matrix& other) =default;
	matrix(matrix&& other)      =default;
    ~matrix()                   =default;


	matrix& operator=(const matrix& other)  =default;
	matrix& operator=(matrix&& other)       =default;

    // Sizing
	size_t size1() const {return cols.empty() ? 0 : cols[0].size(); }
	size_t size2() const {return cols.size();}
	void resize(size_t n, size_t m){
	    cols.resize(m);
	    for(size_t i=0;i<m;++i)cols[i].resize(n);
	}
	void setid(){
        assert(size1()==size2());
	    for(size_t i=0;i<size1();i++){
            SELF(i,i)=1;
            for(size_t j=i+1;j<size1();j++)SELF(i, j)=SELF(j, i)=0;
            }
	    }

    // Indexing
	double get (size_t i, size_t j) {return cols[j][i];}
	void set(size_t i, size_t j, double value){cols[j][i] = value;}
	double& operator()(size_t i, size_t j){return cols[j][i];}
    const double& operator()(size_t i, size_t j) const {return cols[j][i];}
	//double& operator[](size_t i, size_t j){return cols[j][i];}
	//const double& operator[](size_t i, size_t j) const {return cols[j][i];}
	vector& operator[](size_t i){return cols[i];}
	const vector& operator[](size_t i) const {return cols[i];}
//	vector get_col(size_t j);
//	void set_col(size_t j,vector& cj);

    // Transpose
	matrix transpose() const {
        matrix R(size2(),size1());
        for(size_t i=0;i<R.size1();i++)
            for(size_t j=0;j<R.size2();j++) R(i, j)=SELF(j, i);
        return R;
        }

	matrix T() const{return SELF.transpose();}

	matrix& operator+=(const matrix& other) {
        FOR_COLS(i, SELF) SELF[i] += other[i];
        return SELF;}

	matrix& operator-=(const matrix& other) {
        FOR_COLS(i, SELF) SELF[i] -= other[i];
        return SELF;} 

    // Rewrite
	// matrix& operator*=(const matrix& other) {
    //     FOR_COLS(i, SELF) SELF[i] *= other[i];
    //     return SELF;}

	matrix& operator*=(const double x) {
        FOR_COLS(i, SELF) SELF[i] *= x;
        return SELF;}

	matrix& operator/=(const double x) {
        FOR_COLS(i, SELF) SELF[i] /= x;
        return SELF;} 


    long double determinant() const {
    size_t n = SELF.size1();
    size_t m = SELF.size2();
    assert(n == m); // Determinant only exists for square matrices

    matrix temp = SELF;
    long double det = 1.0; 

    for (size_t i = 0; i < n; ++i) {
        // 1. Pivot search: find the largest element in the current COLUMN i
        // We look at rows j = i to n
        size_t pivot_row = i;
        for (size_t j = i + 1; j < n; ++j) {
            if (std::abs(temp(j, i)) > std::abs(temp(pivot_row, i))) {
                pivot_row = j;
            }
        }

        // 2. Swap rows if the best pivot isn't on the diagonal
        if (pivot_row != i) {
            // Because your structure is columns, a "row swap" means
            // swapping elements at index 'i' and 'pivot_row' in EVERY column.
            for (size_t k = 0; k < n; ++k) {
                std::swap(temp(i, k), temp(pivot_row, k));
            }
            det *= -1.0;
        }

        // 3. Singularity check
        if (std::abs(temp(i, i)) < 1e-15) return 0.0;

        // 4. Elimination: Zero out the elements below the pivot in column i
        for (size_t row = i + 1; row < n; ++row) {
            long double factor = temp(row, i) / temp(i, i);
            // Subtract factor * row(i) from row(row)
            // This affects all columns from i+1 to the end
            for (size_t col = i + 1; col < n; ++col) {
                temp(row, col) -= factor * temp(i, col);
            }
        }
        
        // 5. Accumulate the diagonal product
        det *= temp(i, i);
    }

    return det;
    }

    matrix inverse() const {
    size_t n = size1();
    assert(n == size2()); // Must be square

    // 1. Setup Augmented Matrix [temp | result]
    // result starts as Identity
    matrix temp = SELF;
    matrix result(n, n);
    result.setid();

    for (size_t i = 0; i < n; ++i) {
        // 2. Pivot search (for numerical stability)
        size_t pivot = i;
        for (size_t j = i + 1; j < n; ++j) {
            if (std::abs(temp(j, i)) > std::abs(temp(pivot, i))) pivot = j;
        }

        // Swap rows in both temp and result
        if (pivot != i) {
            for (size_t k = 0; k < n; ++k) {
                std::swap(temp(i, k), temp(pivot, k));
                std::swap(result(i, k), result(pivot, k));
            }
        }

        // 3. Scale the pivot row to 1
        double divisor = temp(i, i);
        assert(std::abs(divisor) > 1e-18); // Check for non-invertible matrix

        for (size_t k = 0; k < n; ++k) {
            temp(i, k) /= divisor;
            result(i, k) /= divisor;
        }

        // 4. Eliminate other entries in the column
        for (size_t row = 0; row < n; ++row) {
            if (row != i) {
                double factor = temp(row, i);
                for (size_t k = 0; k < n; ++k) {
                    temp(row, k) -= factor * temp(i, k);
                    result(row, k) -= factor * result(i, k);
                }
            }
        }
    }
    return result;
    }

	void print(std::string s="") const{
        std::cout << s << std::endl;
        for(size_t i=0;i<size1();i++){
            for(size_t j=0;j<size2();j++)printf("%9.3g ",(double)SELF(i, j));
            printf("\n");
            }
        printf("\n");
        }
};

matrix operator/(const matrix& A,double x){
	matrix R=A;
	R/=x;
	return R; }

matrix operator*(const matrix& A,double x){
	matrix R=A;
	R*=x;
	return R; }

matrix operator*(double x,const matrix& A){
	return A*x; }

matrix operator+(const matrix& A, const matrix& B){
	matrix R=A;
	R+=B;
	return R; }

matrix operator-(const matrix& A, const matrix& B){
	matrix R=A;
	R-=B;
	return R; }

vector operator*(const matrix& M, const vector& v){
	vector r; r.resize(M.size1());
	for(size_t i=0;i<r.size();i++){
		double sum=0;
		for(size_t j=0;j<v.size();j++)sum+=M(i, j)*v[j];
		r[i]=sum;
		}
	return r;
	}

matrix operator*(const matrix& A, const matrix& B){
	matrix R(A.size1(),B.size2());
	for(size_t k=0;k<A.size2();k++)
	for(size_t j=0;j<B.size2();j++)
		{
		for(size_t i=0;i<A.size1();i++)R(i, j)+=A(i,k)*B(k,j);
		}
	return R;
	}

matrix pow (const matrix& M, size_t x) {
    matrix R = M;
    FOR_COLS(i, R) R[i] = pow(M[i], x);
    return R;}


std::ostream& operator<<(std::ostream& os, const matrix& M) {
    size_t rows = M.size1();
    size_t cols = M.size2();
    os << std::fixed << std::setprecision(3);
    for (size_t i = 0; i < rows; ++i) {
        os << "[";
        for (size_t j = 0; j < cols; ++j) {
            // Using m(i,j) for row i, column j
            os << std::setw(6) << std::right << M(i, j);
            if (j < cols - 1) os << " ";
        }
        if (i != rows - 1) os << "]\n";
    }
    os << "]";
    return os;}

}
