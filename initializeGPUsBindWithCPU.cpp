#include <sstream>
#include <iostream>
#include <mpi.h>
#include "openacc.h"
#include <string>
#define RED "\033[01;31m"
#define RESET "\033[22;0m"
#define OPENACC 1

using namespace std;

// Jaber Hasbestan, PhD

/*! \brief
 *  code to extract node number from processor name
 *  in other words extracts integers from string
*/

int extractIntegerWords(string str) {
  std::string temp;
  int number = 0;

  for (unsigned int i = 0; i < str.size(); i++) {
    if (isdigit(str[i])) {
      for (unsigned int a = i; a < str.size(); a++) {
        temp += str[a];
      }
      break;
    }
  }
  std::istringstream stream(temp);
  stream >> number;

  return (number);
}

/*! \brief Helper Function to
 *         check one-on-one mapping between CPU-GPU.
 *
 *  1. Processor name is obtained using MPI_Get_processor_name()
 *  2. Node numbers are extracted from the string
 *  3. Local nodal communicator is constricted using MPI_Comm_split()
 *  4. The size of this new communicator is returned to OpenACCINIT
 *  5. The size of the nodl communicator should be equal to number of GPU's
 *  6. If not, error code 1 is thrown
 */

int createNodalCommunicator() {

  MPI_Comm nodalComm;

  int my_rank, np, my_new_rank, new_np;
  int color, key, len;

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  char pname[256];

  MPI_Get_processor_name(pname, &len);

  // some stream  wizardry to extract node number from the proc name
  stringstream ss;
  string s;
  ss << pname;
  ss >> s;
  //   cout<<s<<endl;
  color = extractIntegerWords(s);
  key = my_rank;

  MPI_Comm_split(MPI_COMM_WORLD, color, key, &nodalComm);
  MPI_Comm_rank(nodalComm, &my_new_rank);
  MPI_Comm_size(nodalComm, &new_np);

  cout << "Process" << my_rank << " (in COMM_WORLD): split communicator has "
       << new_np << " processe " << endl;

  MPI_Comm_free(&nodalComm);

  return (new_np);
}

/*! \brief 
 *
 * Perform  CPU-GPU Binding
 *
 */ 


int OPENACC_Init(int &my_rank,  int &com_size)
{
  int result = 1;

#if (OPENACC)
  int nodalComSize;

  acc_init(acc_device_nvidia); // OpenACC call

  const int num_dev = acc_get_num_devices(acc_device_nvidia); // #GPUs

  nodalComSize = createNodalCommunicator();

  cout << nodalComSize << " " << num_dev << endl;

  if ((nodalComSize == num_dev) && (num_dev != 0)) {

    const int dev_id = my_rank % num_dev;

    acc_set_device_num(dev_id, acc_device_nvidia); // assign GPU to one MPI process
    
    cout << "MPI process " << my_rank << "  is assigned to GPU " << dev_id << "\n";
  } 
   else
   {
    cout << RED << " unsuccessful mapping " << RESET << endl;
    exit(1);
  }

#endif
  return (result);
}
// driver
int main(int argc, char *argv[]) {

  int my_rank, np;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  OPENACC_Init(my_rank, np);
  MPI_Finalize();
}
