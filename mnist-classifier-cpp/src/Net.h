#ifndef NET_H
#define NET_H

class Net
{
    LinearConnection fc1(28*28, 1000);
    LinearConnection fc2(1000, 10);
    
    void forward(input);
};

#endif // NET_H
