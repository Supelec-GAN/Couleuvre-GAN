#include "headers/mnist_reader.h"

using namespace std;

mnist_reader::mnist_reader(string pathImage, string pathLabel)
{
    mFullPathImage = pathImage;
    mFullPathLabel = pathLabel;
}


int mnist_reader::reverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}


void mnist_reader::ReadMNIST(vector<Eigen::MatrixXf> &mnist, Eigen::MatrixXi &label)
{
    ifstream file (mFullPathImage,ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);
        mnist.resize(number_of_images);
        for(auto itr(mnist.begin()); itr != mnist.end(); itr++)
        {
            itr->resize(n_rows*n_cols,1);
        }
        for(int i=0;i<number_of_images;++i)
        {
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    mnist[i]((n_rows*r)+c,0) = ((float)temp)/255;
                }
            }
        }
    }
    else
        throw std::runtime_error("mnist_reader::ReadMnist - Unable to open file : " + mFullPathImage);

    ifstream file2 (mFullPathLabel,ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        file2.read((char*)&magic_number,sizeof(magic_number));
        magic_number= reverseInt(magic_number);
        file2.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        label.resize(number_of_images,1);
        for(int i=0;i<number_of_images;++i)
        {
            unsigned char temp=0;
            file2.read((char*)&temp,sizeof(temp));
            label(i,0) = (int)temp;
        }
    }
    else
        throw std::runtime_error("mnist_reader::ReadMnist - Unable to open file : " + mFullPathLabel);
}

