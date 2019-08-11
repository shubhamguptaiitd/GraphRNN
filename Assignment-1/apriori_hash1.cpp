#include<iostream>
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

#include "functions.h"
using namespace std;
using namespace std::chrono; 

string get_string(vector<int> itemset){
    string st = to_string(itemset[0]);
    for(auto &num:itemset){
        st += '#';
        st += to_string(num);
    }
    return st;
}

bool allSubSetFrequent(vector<int> &pattern, vector<vector<int> >& fitemsets,unordered_map<string,int> &mp){
    for (int i = 0; i < pattern.size();i++){
        vector<int> temp(pattern);
        temp.erase(temp.begin()+i);
        long int len = fitemsets.size();
        bool frequent = false;
        if(mp.size() == 0){
            for(int i= 0;i < len ;i++){
                if (temp == fitemsets[i]){
                    frequent = true;
                    break;
                }
            }
        }
        else{
            if(mp.find(get_string(temp)) != mp.end())
                frequent = true;
        }
        if(!frequent)
            return false;
    }
    return true;
}
bool check_if_kitem_in_transaction( unordered_set<int> &transaction,vector<int> &pattern){
    for(int i = 0; i < pattern.size();i++){
        if (transaction.find(pattern[i]) == transaction.end()){
            return false;
        }
    }
    return true;        
}
int main(int argc, char** argv)
{
    auto start = high_resolution_clock::now(); 

    int support_thresh = std::stoi(argv[1]);
    string filename = argv[2];
    FILE *fs = fopen(filename.c_str(),"r");
    size_t pos = filename.find(".");
    string outfilename = filename.substr(0,pos) + ".txt";
    ofstream fout;
    fout.open (outfilename);
    cout << filename << " "<< support_thresh <<"\n";


    /* Program for the apriori algorithm */

    ifstream fin(filename);
    string str_in; 

    /* Computing the support count for items and figuring out 1 frequent items*/
    unordered_map<int, long int > items_map;
    long int num_transactions = 0;
    vector<int> transactions_temp;
    //cout << "came here" ;
    while(fastscan(fs,transactions_temp)){
        
        num_transactions += 1;
//        vector<string> t_str = split(str_in,' ');
        //stringstream str_stream(str_in);
        //for(int num =0;str_stream >> num;){
        for (int num: transactions_temp){
            if(items_map.find(num) == items_map.end()){
                items_map[num] = 1;
            }
            else{
                items_map[num] += 1;
            }            
        }
        transactions_temp.clear();
    }
    cout <<" number of transaction" << num_transactions << endl;
    long int support_ct = ceil((num_transactions*support_thresh)*1.00/100);
    cout << "support threshold " << support_ct << endl;

    vector<vector<int> > k1_itemsets;
    for(auto it = items_map.begin();it != items_map.end(); it++){
        if (it->second >=  support_ct){
            vector<int> v{it->first};
            k1_itemsets.push_back(v);
            fout << it->first << "\n";

        }
        //cout << " : " << it->first << " " << it->second<< endl;
    } 


    cout << "Generated f1 itemsets " << k1_itemsets.size() << endl;
    bool check_if_algo_needs_to_switch_to_hash = true;
    if(support_thresh <= 20 && num_transactions >= 500000 ){
        check_if_algo_needs_to_switch_to_hash = true;
        cout << "Switching to hash mode" << endl;
    }
    get_time_spent(start);

    /* Now create a frequent item sets till possible */
    int k = 2;

    bool create_more_itemsets= true;
    while(create_more_itemsets ){
        cout << "creating itemsets of k lenghth  : " << k << endl;
        cout << "size of k-1 itemset" << k1_itemsets.size() << endl;
        int ct_pruned = 0;int ct_loop_pruned = 0;int ct = 0;
        vector<vector<int> > k_itemsets;
        unordered_map<string,int> fk1_hmp;
        unordered_map<string,int> fk_chmp;
        unordered_set<int> f1_itemsets;
        for(auto &it:k1_itemsets){
            for(auto num:it)
                f1_itemsets.insert(num);
        }
        if(k > 10)
            check_if_algo_needs_to_switch_to_hash = false;

        if(check_if_algo_needs_to_switch_to_hash){
            /* create fk-1 frequent itemset hashmap */
            for(auto &it:k1_itemsets)
                fk1_hmp[get_string(it)] = 1;
        }
        int len = k1_itemsets.size();
        for (int i = 0; i < len;i++){
            
            for (int j= i+1;j < len;j++){
                ct += 1;
                if (equal(k1_itemsets[i].begin(),--k1_itemsets[i].end(),k1_itemsets[j].begin()) && (k1_itemsets[i][k-2] != k1_itemsets[j][k-2])){
                    vector<int> possiblefk(k1_itemsets[i].begin(),--k1_itemsets[i].end());
                    int a = k1_itemsets[i][k-2];
                    int b = k1_itemsets[j][k-2];
                    //cout << "in loop " << a << " "<< b<< endl;
                    //print_vector(possiblefk);

                    if( a < b){
                        possiblefk.push_back(a);possiblefk.push_back(b);
                    }
                    else{
                        possiblefk.push_back(b);possiblefk.push_back(a);
                    }
                    if (allSubSetFrequent(possiblefk,k1_itemsets,fk1_hmp)){ // fk1 hash map will be activated only for condition
                            k_itemsets.push_back(possiblefk);
                    }
                    else
                        ct_pruned++;


                    
                }
                else
                    ct_loop_pruned +=1;
                
            }
        }
        get_time_spent(start);

        cout << "pruned in subset, loop and total " << ct_pruned << " " << ct_loop_pruned << " " << ct << endl;
        cout << "size of possible k candidates extracted: " << k_itemsets.size() << endl;
        //print_vector_vector(k_itemsets);

        if(check_if_algo_needs_to_switch_to_hash){
            /* create fk frequent itemset hashmap */
            cout << "creating hash of itemset" << endl;
            int i = 0;
            for(auto &it:k_itemsets){
                fk_chmp[get_string(it)] = i;
                i++;
            }
        }
        cout << "checking for support count" << endl;

       
        int done = 0;
        FILE *fs = fopen(filename.c_str(),"r");
        vector<int> support_ct_k_freq(k_itemsets.size(),0);
        vector<int> transactions_temp;
        int len_possible_itemsets = k_itemsets.size();
        while((fastscan(fs,transactions_temp)) && (len_possible_itemsets >0)){      
            unordered_set<int> transaction;

            for(int &num:transactions_temp) {
                if(f1_itemsets.find(num)!= f1_itemsets.end())
                    transaction.insert(num);
            }
            for(int i = 0;i <k_itemsets.size();i++){
                    if( check_if_kitem_in_transaction(transaction,k_itemsets[i]))
                    support_ct_k_freq[i] += 1;
                }

            if(done%10000 == 0)
                cout << "done " << done << endl;
            done += 1;
            transactions_temp.clear();

        }
        k1_itemsets.clear();
        cout << " got the count and now cleaning " << endl;
        for(int i = 0;i < k_itemsets.size();i++){
            if(support_ct_k_freq[i] >= support_ct){
                k1_itemsets.push_back(k_itemsets[i]);
                
                /* saving in file  */
                for (auto it :k_itemsets[i])
                    fout << it << " ";
                fout << endl;
            }
        }
        k_itemsets.clear();
        fk_chmp.clear();
        fk1_hmp.clear();
        f1_itemsets.clear();
        k++; // remember to increment this counter , very important one 
        if (k1_itemsets.size() <= 1)
            create_more_itemsets = false;
    }

    cout << "size of k-1 itemset" << k1_itemsets.size() << endl;
    fout.close();
    return 0;
}
