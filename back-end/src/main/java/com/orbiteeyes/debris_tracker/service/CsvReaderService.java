package com.orbiteeyes.debris_tracker.service;

import com.opencsv.CSVReader;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Service;

import java.io.InputStreamReader;
import java.io.Reader;

@Service
@Slf4j
public class CsvReaderService {

    public String[] readLineFromCsv(String satelliteName, int targetLine) throws Exception {
        String filePath = "jury_demo/" + satelliteName + ".csv";
        ClassPathResource resource = new ClassPathResource(filePath);

        try (Reader reader = new InputStreamReader(resource.getInputStream());
             CSVReader csvReader = new CSVReader(reader)) {

            String[] line;
            int currentIndex = 0;
            while ((line = csvReader.readNext()) != null) {
                if (currentIndex == targetLine) {
                    return line;
                }
                currentIndex++;
            }
        }

        log.info("{} için CSV dosyası bitti. (targetLine={})", satelliteName, targetLine);
        return null;
    }
}