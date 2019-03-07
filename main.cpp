#include "Descriptor.h"

/**
 * Main method
 *
 *
 * @param argc number of arguments
 * @param argv arguments
 *
 * @return 0 if everything finished successfully, otherwise 1
 */
int main(int argc, char *argv[]) {
    Descriptor *descriptor = new Descriptor();
    descriptor->start();
    return 0;
}