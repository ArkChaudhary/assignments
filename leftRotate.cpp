#include <iostream>
using namespace std;

void reverse(int arr[], int start, int end) {
    while (start < end) {
        int temp = arr[start];
        arr[start++] = arr[end];
        arr[end--] = temp;
    }
}

void leftRotate(int arr[], int n, int d) {
    d = d % n;
    reverse(arr, 0, d - 1);
    reverse(arr, d, n - 1);
    reverse(arr, 0, n - 1);
}

int main() {
    int n, d;
    cin >> n >> d;
    int arr[n];
    for (int i = 0; i < n; i++) cin >> arr[i];
    leftRotate(arr, n, d);
    for (int i = 0; i < n; i++) cout << arr[i] << " ";
    return 0;
}
