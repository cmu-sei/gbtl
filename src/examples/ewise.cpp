#include <iostream>
#include <grb/grb.hpp>

//****************************************************************************
int main(int argc, char** argv) {
  spec::matrix<int> a({10, 10});
  spec::matrix<int> b({10, 10});
  spec::matrix<int> c({10, 10});
  spec::matrix<int> d({10, 10});

  a[{2, 3}] = 12;
  b[{2, 3}] = 12;

  a[{1, 8}] = 7;
  b[{1, 8}] = 4;

  a[{7, 3}] = 2;
  b[{4, 3}] = 2;

  std::cout << "Matrix: a\n";
  a.printInfo(std::cout);
  std::cout << "Matrix: b\n";
  b.printInfo(std::cout);

  spec::ewise_intersection(c, a, b, spec::times{});

  std::cerr << "Intersection (times) result: \n";
  c.printInfo(std::cout);

  spec::ewise_union(d, a, b, spec::plus{});

  std::cerr << "Union (plus) result: \n";
  d.printInfo(std::cout);
  return 0;
}
