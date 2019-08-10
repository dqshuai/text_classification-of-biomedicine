Each MEDLINE record of the training set (2280) in plain-text format.
Format matches the format that the BioCreative Meta-Server will distribute.


FILE ENCODING
-------------
UTF-8



BCMS/ONLINE COMPETION NOTE
--------------------------
This is exactly the same file format as the BCMS will be sending to online
participants.

FILE STRUCTURE
--
<JournalTitle>?

<ArticleTitle>?

<AuthorList>?

<AbstractText>?

<MeshHeading\n>*

<IdType:ArticleId\n>+
--


FIELD DESCRIPTIONS
------------------
JournalTitle (if any)

line 1 (possibly empty)

PubMed XML "ISOAbbreviation" (preferred) or "JournalTitle" (if no
ISOAbbreviation is given by PubMed) tag content (and an empty line if neither
of the two values exist).
--
ArticleTitle (if any)

line 3 (possibly empty)

PubMed XML "ArticleTitle" tag content.
--
AuthorList, comma-separated (if any)

line 5 (possibly empty)

PubMed XML "Author" tag content, but using only: "<Initials> <LastName>".
--
AbstractText (if any)

line 7 (possibly empty)

PubMed XML "AbstractText" tag content.
--
MeshHeadings (if any)

zero or more lines, starting at line 9+

PubMed XML MeshHeadingList content using the DescriptorName and QualifierName
tag content of MeshHeadings, including the MajorTopicYN attribute value.

Format:
[+-]<DescriptorName> ([+-]<QualifierName>; [+-]<QualifierName>; ...)\n
Where the preceding + or - determines if it is a major or minor topic,
respectively. The qualifiers for a descriptor are listed in parenthesis
(if any; if there are no qualifiers, there are no parenthesis either).
--
IdType:ArticleId (has at least one ID: the PMID)

one or more lines, starting at an undetermined line number

PubMed XML ArticleIdList content using the ArticleID tag content and the
IdType attribute value.

One line is guaranteed in all documents:
pubmed:<PMID [an integer]>

Format:
<IdType>:<ArticleId>\n
