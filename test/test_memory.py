"""
Interactive Memory Test Script

Usage:
    python test_memory_interactive.py

Commands:
    create <obj_name> <text>           - 创建一条记录
    query <content>                    - 查询记录
    list                               - 列出所有记录
    delete <rec_id>                    - 删除记录
    update <rec_id> <obj_name> <text>  - 更新记录
    help                               - 显示帮助
    exit                               - 退出
"""

from src.memory.recorder import Memory
from src.memory.encoder import OpenAIEmbeddingEncoder
from typing import Optional
import os
import openai

os.environ.setdefault("OPENAI_API_KEY", "sk-fcngxSP4xHGTcaWTGHfnK1BHcQsGMuK6qyAmtnEDGtzJOU2m")
os.environ.setdefault("OPENAI_BASE_URL", "https://aigc.x-see.cn/v1")

class InteractiveMemoryTest:
    def __init__(self):
        self.memory = Memory(encoder=OpenAIEmbeddingEncoder())
        self.commands = {
            'create': self.cmd_create,
            'query': self.cmd_query,
            'list': self.cmd_list,
            'delete': self.cmd_delete,
            'update': self.cmd_update,
            'help': self.cmd_help,
            'exit': self.cmd_exit,
        }
        
    def run(self):
        print("=" * 60)
        print("Memory Interactive Test")
        print("=" * 60)
        print("Type 'help' for available commands")
        print()
        
        while True:
            try:
                user_input = input(">>> ").strip()
                if not user_input:
                    continue
                    
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                if cmd in self.commands:
                    self.commands[cmd](args)
                else:
                    print(f"❌ Unknown command: {cmd}")
                    print("Type 'help' for available commands")
                    
            except KeyboardInterrupt:
                print("\n\nBye! 👋")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def cmd_create(self, args: str):
        """
        create <obj_name> <text>
        Example: create apple 红苹果在左边盒子里
        """
        parts = args.split(maxsplit=1)
        if len(parts) < 2:
            print("❌ Usage: create <obj_name> <text>")
            return
        
        obj_name, text = parts
        record = self.memory.create(
            obj_name=obj_name,
            data_type="text",
            data_value=text,
            text=text
        )
        
        print(f"✅ Created record:")
        print(f"   ID: {record.id}")
        print(f"   Object: {record.obj_name}")
        print(f"   Text: {text}")
        print()
    
    def cmd_query(self, args: str):
        """
        query <content>
        Example: query 苹果
        """
        if not args:
            print("❌ Usage: query <content>")
            return
        
        content = args.strip()
        
        print(f"🔍 Querying: '{content}'")
        print("-" * 60)
        
        # 使用统一的 query 接口
        records, scores = self.memory.query(
            content=content,
            obj_threshold=0.75,
            top_k=5
        )
        
        if not records:
            print("❌ No matching records found")
            print()
            return
        
        print(f"✅ Found {len(records)} record(s):\n")
        
        for rec in records:
            score = scores.get(rec.id, 0.0)
            print(f"📄 Record ID: {rec.id}")
            print(f"   Object: {rec.obj_name}")
            print(f"   Text: {rec.data.get('value', 'N/A')}")
            print(f"   Similarity: {score:.4f}")
            print()
    
    def cmd_list(self, args: str):
        """
        list
        """
        records = self.memory.all()
        
        if not records:
            print("📭 No records in memory")
            print()
            return
        
        print(f"📚 Total {len(records)} record(s):")
        print("-" * 60)
        
        for rec in sorted(records, key=lambda r: r.id):
            print(f"ID: {rec.id} | Object: {rec.obj_name:15} | Text: {rec.data.get('value', 'N/A')}")
        
        print()
    
    def cmd_delete(self, args: str):
        """
        delete <rec_id>
        Example: delete 1
        """
        if not args:
            print("❌ Usage: delete <rec_id>")
            return
        
        try:
            rec_id = int(args.strip())
            success = self.memory.delete(rec_id)
            
            if success:
                print(f"✅ Deleted record {rec_id}")
            else:
                print(f"❌ Record {rec_id} not found")
            print()
            
        except ValueError:
            print("❌ Invalid record ID (must be a number)")
            print()
    
    def cmd_update(self, args: str):
        """
        update <rec_id> <obj_name> <text>
        Example: update 1 apple 绿色苹果在右边
        """
        parts = args.split(maxsplit=2)
        if len(parts) < 3:
            print("❌ Usage: update <rec_id> <obj_name> <text>")
            return
        
        try:
            rec_id = int(parts[0])
            obj_name = parts[1]
            text = parts[2]
            
            record = self.memory.update(
                rec_id=rec_id,
                obj_name=obj_name,
                data_value=text,
                text=text
            )
            
            if record:
                print(f"✅ Updated record {rec_id}:")
                print(f"   Object: {record.obj_name}")
                print(f"   Text: {text}")
            else:
                print(f"❌ Record {rec_id} not found")
            print()
            
        except ValueError:
            print("❌ Invalid record ID (must be a number)")
            print()
    
    def cmd_help(self, args: str):
        """Show help"""
        print(__doc__)
    
    def cmd_exit(self, args: str):
        """Exit the program"""
        print("\nBye! 👋")
        exit(0)


def main():
    # 预填充一些测试数据
    tester = InteractiveMemoryTest()
    
    print("📝 Pre-populating some test data...")
    print()
    
    test_data = [
        ("apple", "红苹果在左边盒子里"),
        ("apple", "青苹果在右边盒子里"),
        ("apple", "黄色苹果在桌子上"),
        ("juice", "橙汁在冰箱里"),
        ("juice", "苹果汁在桌上"),
        ("pear", "梨在篮子里"),
        ("banana", "香蕉挂在墙上"),
    ]
    
    for obj_name, text in test_data:
        tester.memory.create(
            obj_name=obj_name,
            data_type="text",
            data_value=text,
            text=text
        )
    
    print(f"✅ Created {len(test_data)} test records")
    print()
    
    # 启动交互式界面
    tester.run()


if __name__ == "__main__":
    main()