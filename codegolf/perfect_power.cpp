#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include "math.h"
#include <chrono>
#define i64 unsigned long long
#define i32 unsigned int
#define f64 double

int main(){
    char* inp = new char[31];
    char** inps = new char*[1000000];
    for (int i = 0; i < 1000000; i++) {
        inps[i] = new char[31];
    }
    int count = 0;
    while(1==scanf("%30s",inp)){
        for (int i = 0; i < 31; i++) {
            inps[count][i] = inp[i];
        }
        count++;
    }
    auto t0 = std::chrono::system_clock::now();
    for (int i = 0; i < count; i++) {
        inp = inps[i];
        // printf("%s = ",inp);
        f64 upper=strtod(inp,NULL);
        //i64 lower=strtoull(inp,NULL,10);
        //i64 lower=atoll(inp);
        //Neither library function produce input mod 2^64 like one would expect, fantabulous!
        i64 lower=0;
        i32 c=0;
        while(inp[c]){
            lower*=10;
            lower+=inp[c]-48;
            c++;
        }
        f64 a,b;
        i64 candidate;
        f64 offvalue;
        f64 offfactor;
        i32 powleft;
        i64 product;
        i64 currentpow;
        for(a=2;a<50;a++){ //Loop for finding large bases (>3).
            b=1.0/a;
            b=pow(upper,b);
            candidate=(i64)(b+.5);
            if(candidate<4){
                break;
            }
            offvalue=b-(f64)candidate;
            offfactor=fabs(offvalue/b);
            if(offfactor<(f64)1e-14){
                product=1;
                powleft=(i32)a;
                currentpow=candidate;
                while(powleft){ //Integer exponentation loop.
                    if(powleft&1){
                        product*=currentpow;
                    }
                    currentpow*=currentpow;
                    powleft=powleft>>1;
                }
                if(product==lower){
                    // printf(" %Iu^%.0lf",candidate,a);
                }
            }
        }
        for(candidate=3;candidate>1;candidate--){ //Loop for finding small bases (<4), 2 cycles of this saves 50 cycles of the previous loop.
            b=log(upper)/log(candidate);
            a=round(b);
            offfactor=fabs(a-b);
            if((offfactor<(f64)1e-14) && (a>1)){
                product=1;
                powleft=(i32)a;
                currentpow=candidate;
                while(powleft){ //Integer exponentation loop.
                    if(powleft&1){
                        product*=currentpow;
                    }
                    currentpow*=currentpow;
                    powleft=powleft>>1;
                }
                if(product==lower){
                    // printf(" %Iu^%.0lf",candidate,a);
                }
            }
        }
        // printf("\n");
        if(inp[0]==113){ //My keyboard lacks an EOF character, so q will have to do.
            return 0;
        }
    }
    auto t1 = std::chrono::system_clock::now();
    printf("%ld\n", std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count());
    return 0;
}