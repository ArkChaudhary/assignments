#include <iostream>
using namespace std;

void moveZeroes(int arr[], int n) {
    int j = 0;
    for (int i = 0; i < n; i++)
        if (arr[i] != 0) arr[j++] = arr[i];
    while (j < n) arr[j++] = 0;
}

int main() {
    int arr[] = {0, 1, 0, 3, 12}, n = 5;
    moveZeroes(arr, n);
    for (int i = 0; i < n; i++) cout << arr[i] << " ";
    return 0;
}
