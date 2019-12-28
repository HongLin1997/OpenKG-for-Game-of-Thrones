cd c:
cd Programs\Apache\apache-jena-fuseki-3.13.1
fuseki-server --loc=store --update /testds

rm .\store\*
echo "clean store folder"