FROM myllmbox/qwen38-flash-next-cluster-vllm@sha256:a87154e4edca649013b7b8934df70ac53e9f096f5d0e42a823f0ee36452ca315
COPY overlay/ /usr/local/lib/python3.12/dist-packages/
COPY mbx_opt_quant.py.patch /opt/thinkingcap/mbx_opt_quant.py.patch
COPY licenses/ /usr/share/licenses/thinkingcap/
COPY verify_overlay.py overlay-lock.json source-lock.json /opt/thinkingcap/
RUN cd /usr/local/lib/python3.12/dist-packages && patch --batch --fuzz=0 -p1 < /opt/thinkingcap/mbx_opt_quant.py.patch \
    && python3 /opt/thinkingcap/verify_overlay.py --installed /usr/local/lib/python3.12/dist-packages \
    && python3 -c "import hashlib,json,pathlib; lock=json.loads(pathlib.Path('/opt/thinkingcap/source-lock.json').read_text()); assert hashlib.sha256(pathlib.Path('mbx_opt_quant.py').read_bytes()).hexdigest()==lock['mbx_opt_quant_installed_sha256']"
LABEL org.thinkingcap.profile="thinkingcap" \
      org.thinkingcap.overlay="e33f58a4c50bbcd16b7a8b4b115bdac3f7055b62f65bebcd54b7b6c4d4601a41"
