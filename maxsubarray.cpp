#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

vector<int> maxsubarray(vector<int> &arr){
    sort(arr.begin(), arr.end());
    for(int i = 0; i<arr.size(); i++){
        if(arr[i]>0)return vector<int>(arr.begin() + i, arr.end());
    }
    return {};
}

int main(){
    vector<int> array(50);
    vector<int> maxsub(50);
    int n, x;
    cout<<"Enter the length of array";
    cin>>n;
    for(int i = 0; i<n ; i++){
        cin>>x;
        array.push_back(x);
    }
    maxsub = maxsubarray(array);
    for(int i = 0; i<maxsub.size(); i++){
        cout<<maxsub[i]<<" ";
    }
    cout<<endl;
    return 0;
}
