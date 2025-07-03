#include <iostream>
using namespace std;

void reverseIter(int arr[], int n) {
    int i = 0, j = n - 1, temp;
    while (i < j) {
        temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
        i++;
        j--;
    }
}

void reverseRec(int arr[], int start, int end) {
    if (start >= end) return;
    int temp = arr[start];
    arr[start] = arr[end];
    arr[end] = temp;
    reverseRec(arr, start + 1, end - 1);
}

int main() {
    int arr[] = {1, 2, 3, 4, 5}, n = 5;
    reverseIter(arr, n);
    for (int i = 0; i < n; i++) cout << arr[i] << " ";
  
    reverseRec(arr, 0, n - 1);
    for (int i = 0; i < n; i++) cout << arr[i] << " ";
    return 0;
}

