FROM ghcr.io/music-assistant/server:latest

COPY aec_reference.py aec_reference.patch apply_aec_reference_patch.py /tmp/aec-reference/
RUN /app/venv/bin/python /tmp/aec-reference/apply_aec_reference_patch.py \
    && rm -rf /tmp/aec-reference

COPY apply_spotify_daylist_patch.py /tmp/apply_spotify_daylist_patch.py
RUN /app/venv/bin/python /tmp/apply_spotify_daylist_patch.py \
    && rm /tmp/apply_spotify_daylist_patch.py

COPY airplay_mdns_launcher.py /usr/local/bin/
ENTRYPOINT ["/app/venv/bin/python", "/usr/local/bin/airplay_mdns_launcher.py"]
CMD ["/usr/local/bin/entrypoint.sh", "--data-dir", "/data", "--cache-dir", "/data/.cache"]
