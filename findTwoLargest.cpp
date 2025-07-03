#include <iostream>
using namespace std;

void findTwoLargest(int arr[], int n) {
    int first = -1, second = -1;
    for (int i = 0; i < n; i++) {
        if (arr[i] > first) {
            second = first;
            first = arr[i];
        } else if (arr[i] > second && arr[i] != first) {
            second = arr[i];
        }
    }
    cout << first << " " << second;
}

int main() {
    int arr[] = {4, 2, 9, 1, 9, 5}, n = 6;
    findTwoLargest(arr, n);
    return 0;
}
