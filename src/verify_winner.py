import json

def filter_different_winners(input_file, output_file):
    """
    Tách ra những dòng có winner_before khác với winner
    
    Args:
        input_file: Đường dẫn file JSONL đầu vào
        output_file: Đường dẫn file JSONL đầu ra
    """
    count = 0
    total = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            total += 1
            data = json.loads(line.strip())
            
            # Kiểm tra nếu winner_before khác winner
            if data.get('winner_before') != data.get('winner'):
                outfile.write(line)
                count += 1
    
    print(f"Tổng số dòng: {total}")
    print(f"Số dòng có winner_before khác winner: {count}")
    print(f"Tỷ lệ: {count/total*100:.2f}%")
    print(f"\nĐã lưu kết quả vào: {output_file}")


if __name__ == "__main__":
    # Cấu hình đường dẫn
    input_file = 'src/verify_response/Llama-2-7b-chat-hf/vicuna_eval/ori_bpo.jsonl'
    output_file = 'src/verify_winner/Llama-2-7b-chat-hf/vicuna_eval/ori_bpo.jsonl'
    
    # Thực hiện lọc
    filter_different_winners(input_file, output_file)