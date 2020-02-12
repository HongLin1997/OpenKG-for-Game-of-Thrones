cd c:
cd \Programs\Apache\apache-jena-fuseki-3.13.1
rm .\store\*
echo "clean store folder"
fuseki-server --loc=store --update /testds

