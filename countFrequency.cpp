#include <iostream>
using namespace std;

int main() {
    int arr[] = {1, 2, 2, 3, 1, 4}, n = 6;
    bool counted[100] = {false};

    for (int i = 0; i < n; i++) {
        if (counted[i]) continue;
        int count = 1;
        for (int j = i + 1; j < n; j++) {
            if (arr[i] == arr[j]) {
                counted[j] = true;
                count++;
            }
        }
        cout << arr[i] << " - " << count << endl;
    }
    return 0;
}
