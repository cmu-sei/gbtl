#include <grb/grb.hpp>

int main(int argc, char** argv) {
  spec::matrix<int> m({100, 100});

  // Write to missing element.
  m[{4, 4}] = 12;

  // Access present element.
  int v = m[{4, 4}];
  std::cout << v << std::endl;

  // Access missing element.
  int g = m[{4, 3}];
  std::cout << g << std::endl;

  // Write to present element.
  m[{4, 3}] = 12;

  g = m[{4, 3}];
  std::cout << g << std::endl;


  return 0;
}
