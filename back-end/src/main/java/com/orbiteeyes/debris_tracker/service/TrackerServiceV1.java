package com.orbiteeyes.debris_tracker.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
@Slf4j
@RequiredArgsConstructor
public class TrackerServiceV1 {

    private final CsvReaderService csvReaderService;


    private final String[] activeSatellites = {
            "IMECE", "GOKTURK-1", "GOKTURK-2", "GOKTURK-3",
            "TURKSAT-3A", "TURKSAT-4A", "TURKSAT-4B",
            "TURKSAT-5A", "TURKSAT-5B", "TURKSAT-6A", "BILSAT", "RASAT"
    };

    public List<Map<String, Object>> processAllSatellites() {
        List<Map<String, Object>> responseList = new ArrayList<>();

        for (String satName : activeSatellites) {

            int randomLine = (int) (Math.random() * 220) + 1;

            try {
                String[] rawData = csvReaderService.readLineFromCsv(satName, randomLine);

                if (rawData != null && rawData.length >= 38) {
                    Map<String, Object> frontendData = new HashMap<>();

                    frontendData.put("eventId", satName + "-EVT-" + randomLine);

                    frontendData.put("f01TimeToTca", parseSafe(rawData[0]));
                    frontendData.put("f02SatX", parseSafe(rawData[1]));
                    frontendData.put("riskScore", parseSafe(rawData[2]));
                    frontendData.put("status", rawData[3]);
                    frontendData.put("isCollision", rawData[4]);

                    for (int i = 5; i <= 38; i++) {
                        if (i < rawData.length) {
                            frontendData.put("feat_" + i, parseSafe(rawData[i]));
                        } else {
                            frontendData.put("feat_" + i, 0.0);
                        }
                    }

                    responseList.add(frontendData);
                }

            } catch (Exception e) {
                log.error("❌ {} uydusu işlenirken hata: {}", satName, e.getMessage());
            }
        }

        return responseList;
    }

    private double parseSafe(String val) {
        try {
            return Math.round(Double.parseDouble(val) * 10000.0) / 10000.0;
        } catch (Exception e) {
            return 0.0;
        }
    }
}