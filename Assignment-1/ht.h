#ifndef ht_h
#define ht_h

#include<iostream>
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

//int max_leaf_nodes = 3;
struct htnode{
    bool isLeaf= false;
    vector<vector<int>> itemsets;
    vector<long int> positions_in_count_array;
    vector<htnode*> sons;
    unordered_map<int,int> hash_maps;
};
class ht{
    public:
        htnode* root;
        int depth;
        int breadth_nodes;
        ht(int depth,int breath_nodes){
            this->depth = depth;
            this->root = new htnode;
            this->root->isLeaf = false;
            //vector<htnode*> temp(BREADTH_NODES,NULL);
            //root.sons = temp;
            this->breadth_nodes = breath_nodes;
        }


        htnode* add_itemset(vector<int> itemset,int index,long int count_index,htnode *node){

            if(index == this->depth){
                //cout << "cam here 3" << endl;
                node->itemsets.push_back(itemset);
                node->positions_in_count_array.push_back(count_index);
                node->isLeaf = true;
                return node;
            }
            // cout << "chame here 4\n";
            int which_son = itemset[index]%(this->breadth_nodes);
            // cout << which_son << endl;
            // cout << node->hash_maps[which_son] << endl;
            if(node->hash_maps.find(which_son) == node->hash_maps.end()){
                htnode *node_son = new htnode;
                node_son->isLeaf = false;
                node->hash_maps[which_son] = node->sons.size();
                node->sons.push_back(node_son);
            }
            node->sons[node->hash_maps[which_son]] = add_itemset(itemset,index+1,count_index,node->sons[node->hash_maps[which_son]]);

            return node;
        }

        void print_ht(htnode *node){
            if (node== NULL){
                return;
            }
            if(node->isLeaf){
                int ct=0;
                for(auto &it:node->itemsets){
                    print_vector(it);
                    //cout << "position in actual " << node->positions_in_count_array[ct] << endl;
                }
                cout << "--"<< endl;
            }
            else{
                for(auto& it:node->sons)
                    print_ht(it);
            }
            return;
        }
        int ht_search(vector<int> &itemset,int index,htnode* node){
            // if(node == NULL){
            //     cout <<" test -1 " <<endl;
            //     return -1;
            // }
            if(index==this->depth){
                if(node->isLeaf){
                    int ct = 0;
                    for(auto &item: node->itemsets){
                        // cout << "--";
                        // print_vector(item);
                        // cout <<"--";
                        if(std::equal(itemset.begin(),itemset.end(),item.begin()))
                            return node->positions_in_count_array[ct];
                        ct++;
                    }
                }
                //cout << "came here " << endl;
                return -1;
            }
            int which_son = itemset[index]%(this->breadth_nodes);
            //cout <<"test - " << index << " " << which_son <<" "<< itemset[index] << endl;
            return ht_search(itemset,index+1,node->sons[node->hash_maps[which_son]]);


        }
        void insert(vector<int> itemset,long int count_index){
            //int which_son = itemset[0]%(this->breadth_nodes);
            //this->root.sons[which_son]= add_itemset(itemset,1,count_index,this->root.sons[which_son]);
            this->root = add_itemset(itemset,0,count_index,this->root);
            return;
        }
        int search(vector<int> &itemset){
            //int which_son = itemset[0]%(this->breadth_nodes);
            //return ht_search(itemset,1,this->root.sons[which_son]);
            return ht_search(itemset,0,this->root);
        }
        void print(){
            //for(int i = 0;i <this->breadth_nodes;i++ ){
            //    print_ht(this->root.sons[i]);
            //}
            print_ht(this->root);
        }
    
};




#endif
