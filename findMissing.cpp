#include <iostream>
using namespace std;

int main() {
    int n;
    cin >> n;
    int arr[n - 1], sum = n * (n + 1) / 2, actual = 0;
    for (int i = 0; i < n - 1; i++) {
        cin >> arr[i];
        actual += arr[i];
    }
    cout << sum - actual;
    return 0;
}
