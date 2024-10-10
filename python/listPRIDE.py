import requests
import csv

url = 'https://www.ebi.ac.uk/pride/ws/archive/v2/projects?'

def get_all_paged_files(page_size, page, sort_direction, sort_conditions):
        """
         Get all filtered pride submission files
        :param query_filter: Parameters to filter the search results
        :param page_size: Number of results to fetch in a page
        :param page: Identifies which page of results to fetch
        :param sort_direction: Sorting direction: ASC or DESC
        :param sort_conditions: Field(s) for sorting the results on
        :return: paged file list on JSON format
        """
        """
           
        """
        request_url = url
        request_url += "pageSize=" + str(page_size) + "&page=" + str(page) + "&sortDirection=" + sort_direction + "&sortConditions=" + sort_conditions

        return requests.get(request_url).json()




year2writer = dict()

for year in range(2000,2014):
    csvfile = open("/Users/dennisgoldfarb/Downloads/PRIDE/PRIDE_" + str(year) + ".tsv", 'w', newline="\n")
    year2writer[year] = csv.writer(csvfile, delimiter="\t")
    # header
    year2writer[year].writerow(["Accession", "FTP", "Database", "Enzyme", "Include", "Exclude", "Instruments", "ExpTypes", "Organisms", "OrganismParts", "PTMs", "Quant", "Title", "SubmissionDate", "sampleProcessingProtocol", "dataProcessingProtocol"])



for page in range(274,400):
    print("Page", page)
    response = get_all_paged_files(100, page, "DESC", "submissionDate")
   

    for x in response['_embedded']["projects"]:
        FTP_loc = ""
        for attr in x["additionalAttributes"]:
            if attr["name"] == "Dataset FTP location":
                if "generated" in attr["value"]:
                    FTP_loc = "/".join(attr["value"].split("/")[-4:-1])
                else:
                    FTP_loc = "/".join(attr["value"].split("/")[-3:])
                
        instruments = ";".join(v["name"] for v in x["instruments"])
        organisms = ";".join(v["name"] for v in x["organisms"])
        expTypes = ';'.join(v["name"] for v in x["experimentTypes"])
        organismParts = ';'.join(v["name"] for v in x["organismParts"])
        PTMs = ';'.join(v["name"] for v in x["identifiedPTMStrings"])
        quant = ';'.join(v["name"] for v in x["quantificationMethods"])
        year = int(x["submissionDate"].split("-")[0])
        if year >= 2014: continue
        
        year2writer[year].writerow([x["accession"], FTP_loc, "", "", "", "", 
                                    instruments, expTypes, organisms, organismParts, PTMs, quant,
                                    x["title"], x["submissionDate"], 
                                    x["sampleProcessingProtocol"].replace("\r", "").replace("\n", ""), 
                                    x["dataProcessingProtocol"].replace("\r", "").replace("\n", "")])

