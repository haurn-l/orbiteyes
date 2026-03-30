package com.orbiteeyes.debris_tracker.controller;

import com.orbiteeyes.debris_tracker.service.TrackerServiceV1;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/v1/satellites")
@RequiredArgsConstructor
@Slf4j
@CrossOrigin(origins = "*")
public class TrackerController {
    private final TrackerServiceV1 trackerServiceV1;

    @GetMapping("/all")
    public ResponseEntity<List<Map<String, Object>>> getRealTimeTelemetry() {
        List<Map<String, Object>> currentData = trackerServiceV1.processAllSatellites();
        return ResponseEntity.ok(currentData);
    }
}