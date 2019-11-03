#!/bin/bash

lines=(`cat $1 | grep -n "X:" | awk '{split($0,c,":"); print c[1]}'`)
for ii in `seq 0 $((${#lines[@]}-2))`; do
  echo $(($ii+1)) "of" $((${#lines[@]}-1))
  commd=`echo ${lines[$ii]},$((lines[$((ii+1))]-1))p";"$((lines[$((ii+1))]-1))q`
  sed -n $commd $1 | sed '/^$/d' > tempabc_0
  cat tempabc_0 | sed 's/ //g'
  cat tempabc_0 | sed '/^[XTKM]/d' | sed 's/[ |:(\/><]//g' | sed 's/[1-9]//g' |
  sed 's/=C/1 /g' | sed 's/\^C/2 /g' | sed 's/=D/3 /g' | sed 's/\^D/4 /g' | sed 's/=E/5 /g' | sed 's/=F/6 /g' | 
  sed 's/\^F/7 /g' | sed 's/=G/8 /g' | sed 's/\^G/9 /g' | sed 's/=A/10 /g' | sed 's/\^A/11 /g' | sed 's/=B/12 /g' |
  sed 's/=c/13 /g' | sed 's/\^c/14 /g' | sed 's/=d/15 /g' | sed 's/\^d/16 /g' | sed 's/=e/17 /g' | sed 's/=f/18 /g' |
  sed 's/z//g' |
  awk '{ 
   split($0,a," "); 
   printf a[1] " ";
   counter = 1;
   for (i=2; i<=length(a); i++) {
     if(a[i] < 0){
       if((a[i] != -a[i-1]) && (a[i]!=a[i-1])){printf -a[i] " "; counter++}
     } else {printf a[i] " "; counter++}
   if(counter==12){printf "\n"; counter=0;}
   }
   printf "\n"
  }' > tempabc_1

  cat tempabc_1 |
  awk '{
   split($0,a," "); 
   counter = 1;
   for (i=2; i<=length(a); i++) {
     printf a[i]-a[i-1] " ";
     counter++;
     if(counter==12){printf ""; counter=0;}
   }
   printf "\n"
  }' > tempabc_2
  echo
  cat tempabc_1 | awk '{
    split($0,a," ");
   printf a[1] " ";
   for (i=2; i<=length(a); i++) {
     if (a[i] > 12) printf a[i]-12 " "; else printf a[i] " ";
   }
   printf "\n"
  }'
  cat tempabc_2
done
