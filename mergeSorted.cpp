#include <iostream>
using namespace std;

int main() {
    int n1, n2;
    cin >> n1;
    int a[n1];
    for (int i = 0; i < n1; i++) cin >> a[i];

    cin >> n2;
    int b[n2];
    for (int i = 0; i < n2; i++) cin >> b[i];

    int merged[n1 + n2], i = 0, j = 0, k = 0;

    while (i < n1 && j < n2)
        merged[k++] = (a[i] < b[j]) ? a[i++] : b[j++];

    while (i < n1) merged[k++] = a[i++];
    while (j < n2) merged[k++] = b[j++];

    for (int m = 0; m < n1 + n2; m++) cout << merged[m] << " ";
    return 0;
}
