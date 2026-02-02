# Tao de online (MVP)

Web app tao cau hoi tu mau (paraphrase, thay so, doi ngu canh) va xuat file.

## Chay nhanh

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Mo trinh duyet: http://localhost:8000

## Trien khai Vercel

1) Cai Vercel CLI va dang nhap

```bash
npm i -g vercel
vercel login
```

2) Deploy

```bash
vercel
```

3) (Tuy chon) set bien moi truong

```bash
vercel env add OPENAI_API_KEY
vercel env add OPENAI_MODEL
```

File can thiet da co:
- `api/index.py` (entrypoint)
- `vercel.json`

## Tinh nang

- Nhap cau hoi mau (moi dong 1 cau)
- Chon chu de + tu khoa tuy chon
- Paraphrase (tu dong thay tu dong nghia co ban)
- Thay so lieu (tu dong doi so)
- Doi ngu canh (tu dong thay cum tu theo chu de)
- Thu vien mau cau hoi theo mon (chon de chen nhanh)
- Xuat TXT, CSV, DOCX, PDF
- Tuy chon dung AI (OpenAI) neu co khoa API

## Cau hinh AI (tuy chon)

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-4o-mini"
```

Mac dinh se su dung OpenAI Responses API. Neu khong co khoa, he thong tu dong quay ve cach sinh cau hoi co ban.

## Ghi chu ky thuat

- Logic sinh cau hoi nam trong `app/main.py`.
- Danh sach chu de/synonyms/templates co the mo rong trong `TOPICS`, `SYNONYMS`, `TEMPLATES`.
- PDF co wrap dong va co thu font tieng Viet (neu he thong co font phu hop).

## Huong trien khai desktop (goi y)

1) Tauri: dung web frontend nhu hien tai, goi den backend FastAPI qua localhost.
2) Electron: dong goi frontend + chay backend FastAPI kem theo.
3) Python + WebView: dung pywebview/mo webview mo trang web local.

Neu muon, minh co the:
- them template cau hoi theo mon hoc
- them logic paraphrase tieng Viet nang cao
- dong goi thanh desktop app
