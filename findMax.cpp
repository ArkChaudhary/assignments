#include <iostream>
using namespace std;

int findMax(int arr[], int n) {
    int max = arr[0];
    for (int i = 1; i < n; i++)
        if (arr[i] > max) max = arr[i];
    return max;
}

int main() {
    int arr[] = {3, 7, 2, 9, 4}, n = 5;
    cout << findMax(arr, n);
    return 0;
}
