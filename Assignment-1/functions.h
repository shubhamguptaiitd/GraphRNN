#ifndef functions_h
#define functions_h

#include<iostream>
#include<algorithm>
#include <fstream>
#include <string>  
#include <vector> 
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include<cmath>
#include<string>
#include<sstream>
#include <chrono> 
#include<cstdio>
//#include "functions.h"
using namespace std::chrono; 
using namespace std;

//https://www.geeksforgeeks.org/fast-io-for-competitive-programming/
int fastscan(FILE *fs,vector<int>& transaction) 
{ 
    register int c; 
    int number = 0; 
    c = getc_unlocked(fs); 
    if(c==EOF)
        return 0;
    bool begin = true;
    while(begin){
        bool went_to_loop = false;
        for (; (c>47 && c<58); c=getc_unlocked(fs)){ 
        went_to_loop = true;
        number = number *10 + c - 48;
        }
    if(went_to_loop)
        transaction.push_back(number);
     if(c == '\n'){
        begin = false;
        return 1;
    }
    if(c==EOF)
        return 1 ;
    c = getc_unlocked(fs);    
    number =0;
    }    
    return 1;

} 
  

vector<vector<int>> generate_per(vector<int> &vec,int k){
    vector<vector<int>>  subsets;
    if (k > vec.size())
        return subsets;

    vector<bool> v(vec.size());
    fill(v.begin(), v.begin() + k, true);

    do {
       vector<int> temp(k);
        int ct=0;
       for (int i = 0; i < vec.size(); ++i) {
           if (v[i]) {
               temp[ct] = vec[i];
               ct+= 1;
           }
          
       }
        subsets.push_back(temp);
   } while (std::prev_permutation(v.begin(), v.end()));

   return subsets;
}

void print_set(set<int> const &s)
{
    for (auto const& i: s)
        cout << i << " ";
    cout << "\n";
}

void get_time_spent(high_resolution_clock::time_point start){

    auto stop = high_resolution_clock::now(); 
    auto duration = duration_cast<microseconds>(stop - start); 
    cout << "Time taken by function: "<< duration.count()*1.00/1000000 << " seconds" << endl; 
    cout << "Time taken by function: "<< duration.count()*1.00/60000000 << " minutes" << endl;

}

vector<string> split(string str, const char delim)
{
    vector<string> substrs;
    std::stringstream ss(str);
	string s;
	while (getline(ss, s, delim)) {
		substrs.push_back(s);
	}

    return substrs;
}

void print_vector_set(vector<set<int> >const &s)
{
    for (auto const& i: s){
        for(auto const&j : i)
            cout << j << " ";
        cout << "\n";
    }
}

void print_vector(vector<int> t){
    for(int i = 0 ; i < t.size();i++){
        cout << t[i] << " ";
    }
    cout << "\n";

}

void print_long_vector(vector<long int> t){
    for(int i = 0 ; i < t.size();i++){
        cout << t[i] << " ";
    }
    cout << "\n";

}
void print_vector_vector(vector<vector<int> > t){
    for(int i = 0 ; i < t.size();i++){
        for (int j = 0; j < t[i].size();j++){
            cout << t[i][j] << " ";

        }
        cout << "\n\n";

    }

}
#endif
