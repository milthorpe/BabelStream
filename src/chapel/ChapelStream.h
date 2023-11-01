#include <string>
extern "C" {
int get_device_driver_version(const int device);
std::string get_device_name(const int device);
}