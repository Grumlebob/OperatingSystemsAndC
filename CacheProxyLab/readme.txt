make clean
make
./proxy 20103
Or:
make clean && make && ./proxy 20103
make clean && make && ./proxy 20103 >> output.txt    (Den tager åbentbart ikke det hele med)
And then in other terminal:
curl -v --proxy http://cos.itu.dk:20103/ http://www.example.com
curl -v --proxy http://cos.itu.dk:20103/ http://www.testingmcafeesites.com/



23-PxeDriver tut:
1. Make clean && make   //Inside this repo
2. mv proxy ../23-pxedrive/     //Move exe to pxedrive folder
3. inside 23-PxeDriver folder (other terminal is easier) pxy/pxydrive.py -p ./proxy -f s03-overrun.cmd
