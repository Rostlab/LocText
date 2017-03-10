* Collect the SwissProt localization relations from (show columns ac, organism id, and GO CC, and then download):

http://www.uniprot.org/uniprot/?query=reviewed%3Ayes+AND+%28taxonomy%3A40559+OR+taxonomy%3A9986+OR+taxonomy%3A7955+OR+taxonomy%3A4039+OR+taxonomy%3A4081+OR+taxonomy%3A6239+OR+taxonomy%3A4679+OR+taxonomy%3A4787+OR+taxonomy%3A9913+OR+taxonomy%3A3885+OR+taxonomy%3A10116+OR+taxonomy%3A3888+OR+taxonomy%3A4072+OR+taxonomy%3A7227+OR+taxonomy%3A4577+OR+taxonomy%3A562+OR+taxonomy%3A4097+OR+taxonomy%3A10090+OR+taxonomy%3A3702+OR+taxonomy%3A4932+OR+taxonomy%3A9606%29&sort=score

That is:

```
reviewed:yes AND (taxonomy:40559 OR taxonomy:9986 OR taxonomy:7955 OR taxonomy:4039 OR taxonomy:4081 OR taxonomy:6239 OR taxonomy:4679 OR taxonomy:4787 OR taxonomy:9913 OR taxonomy:3885 OR taxonomy:10116 OR taxonomy:3888 OR taxonomy:4072 OR taxonomy:7227 OR taxonomy:4577 OR taxonomy:562 OR taxonomy:4097 OR taxonomy:10090 OR taxonomy:3702 OR taxonomy:4932 OR taxonomy:9606)
```
