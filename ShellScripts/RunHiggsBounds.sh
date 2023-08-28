#!/bin/bash

cd $1

prefix='../../SPheno-4.0.5/'

#nH='3'
#nHplus='1'

nH=3
nHplus=1

whichinput='effC'   # whichinput can be 'part', 'hadr' or 'effC'
whichanalyses='LandH'   # whichanalyses can be 'LandH', 'onlyL', 'onlyH' or 'onlyP' 

# echo ./HiggsBounds $whichanalyses $whichinput $nH $nHplus $prefix
./HiggsBounds $whichanalyses $whichinput $nH $nHplus $prefix &> /dev/null

#echo ' **************************************************'
#echo ' The output files are'
#echo ' '"$prefix"HiggsBounds_results.dat
#echo " and"
#echo ' '"$prefix"Key.dat
#echo ' **************************************************'
