import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from wrs.drivers.devices.chassis.chassis1 import control_chassis
class ChassisControlTester:
    def __init__(self, root):
        self.root = root
        self.root.title("底盘控制测试系统")
        self.root.geometry("800x600")
        
        # 创建主框架
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 控制变量
        self.running = False
        self.stop_requested = False
        self.control_thread = None
        
        # 创建控制面板
        self.create_control_panel()
        
        # 创建状态显示
        self.create_status_display()
        
        # 创建按钮区域
        self.create_button_area()
        
        # 初始化参数
        self.reset_parameters()
    
    def create_control_panel(self):
        # 控制参数面板
        control_panel = ttk.LabelFrame(self.main_frame, text="控制参数", padding="10")
        control_panel.pack(fill=tk.X, pady=5)
        
        # 运行时间
        ttk.Label(control_panel, text="运行时间(秒):").grid(row=0, column=0, sticky=tk.W)
        self.run_time_entry = ttk.Entry(control_panel)
        self.run_time_entry.grid(row=0, column=1, sticky=tk.EW)
        self.run_time_entry.insert(0, "5.0")
        
        # 水泵开关
        self.water_pump_var = tk.BooleanVar()
        ttk.Checkbutton(control_panel, text="水泵开关", variable=self.water_pump_var).grid(row=1, column=0, sticky=tk.W)
        
        # 远程使能
        self.remote_enable_var = tk.BooleanVar()
        ttk.Checkbutton(control_panel, text="远程使能", variable=self.remote_enable_var).grid(row=1, column=1, sticky=tk.W)
        
        # 档位位置
        ttk.Label(control_panel, text="档位位置:").grid(row=2, column=0, sticky=tk.W)
        self.gear_position_var = tk.StringVar()
        gear_combobox = ttk.Combobox(control_panel, textvariable=self.gear_position_var, 
                                    values=["0:无效", "1:R档", "2:N档", "3:D档"], state="readonly")
        gear_combobox.grid(row=2, column=1, sticky=tk.EW)
        gear_combobox.current(2)  # 默认N档
        
        # 目标速度
        ttk.Label(control_panel, text="目标速度(m/s):").grid(row=3, column=0, sticky=tk.W)
        self.target_speed_scale = ttk.Scale(control_panel, from_=0, to=10, orient=tk.HORIZONTAL)
        self.target_speed_scale.grid(row=3, column=1, sticky=tk.EW)
        self.target_speed_var = tk.StringVar()
        self.target_speed_var.set("0.0")
        ttk.Label(control_panel, textvariable=self.target_speed_var).grid(row=3, column=2, sticky=tk.W)
        self.target_speed_scale.bind("<Motion>", lambda e: self.target_speed_var.set(f"{self.target_speed_scale.get():.1f}"))
        
        # 底盘模式
        ttk.Label(control_panel, text="底盘模式:").grid(row=4, column=0, sticky=tk.W)
        self.chassis_mode_var = tk.StringVar()
        mode_combobox = ttk.Combobox(control_panel, textvariable=self.chassis_mode_var, 
                                    values=["0:无效", "1:直行", "2:前驱阿克曼", "3:斜移", 
                                           "4:自旋", "5:小转弯", "6:横移", "7:驻车"], state="readonly")
        mode_combobox.grid(row=4, column=1, sticky=tk.EW)
        mode_combobox.current(7)  # 默认驻车
        
        # 目标角度
        ttk.Label(control_panel, text="目标角度(度):").grid(row=5, column=0, sticky=tk.W)
        self.target_angle_scale = ttk.Scale(control_panel, from_=-180, to=180, orient=tk.HORIZONTAL)
        self.target_angle_scale.grid(row=5, column=1, sticky=tk.EW)
        self.target_angle_var = tk.StringVar()
        self.target_angle_var.set("0")
        ttk.Label(control_panel, textvariable=self.target_angle_var).grid(row=5, column=2, sticky=tk.W)
        self.target_angle_scale.bind("<Motion>", lambda e: self.target_angle_var.set(f"{self.target_angle_scale.get():.0f}"))
        
        # 使所有列可扩展
        for i in range(3):
            control_panel.columnconfigure(i, weight=1)
    
    def create_status_display(self):
        # 状态显示面板
        status_panel = ttk.LabelFrame(self.main_frame, text="状态信息", padding="10")
        status_panel.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.status_text = tk.Text(status_panel, height=10, state=tk.DISABLED)
        self.status_text.pack(fill=tk.BOTH, expand=True)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(status_panel, orient=tk.VERTICAL, command=self.status_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text.configure(yscrollcommand=scrollbar.set)
    
    def create_button_area(self):
        # 按钮面板
        button_panel = ttk.Frame(self.main_frame)
        button_panel.pack(fill=tk.X, pady=5)
        
        # 开始按钮
        self.start_button = ttk.Button(button_panel, text="开始控制", command=self.start_control)
        self.start_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        # 紧急停止按钮
        self.stop_button = ttk.Button(button_panel, text="紧急停止", command=self.emergency_stop, 
                                    style="Emergency.TButton")
        self.stop_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        self.stop_button.state(['disabled'])
        
        # 重置按钮
        ttk.Button(button_panel, text="重置参数", command=self.reset_parameters).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        # 退出按钮
        ttk.Button(button_panel, text="退出", command=self.root.quit).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        # 配置紧急停止按钮样式
        style = ttk.Style()
        style.configure("Emergency.TButton", foreground="white", background="red", font=('Helvetica', 12, 'bold'))
    
    def reset_parameters(self):
        """重置所有参数到安全默认值"""
        self.run_time_entry.delete(0, tk.END)
        self.run_time_entry.insert(0, "5.0")
        self.water_pump_var.set(False)
        self.remote_enable_var.set(False)
        self.gear_position_var.set("2:N档")
        self.target_speed_scale.set(0)
        self.target_speed_var.set("0.0")
        self.chassis_mode_var.set("7:驻车")
        self.target_angle_scale.set(0)
        self.target_angle_var.set("0")
        
        self.append_status("参数已重置为安全默认值")
    
    def start_control(self):
        """开始控制底盘"""
        if self.running:
            messagebox.showwarning("警告", "控制已在运行中!")
            return
        
        try:
            # 获取参数
            run_time = float(self.run_time_entry.get())
            water_pump = self.water_pump_var.get()
            remote_enable = self.remote_enable_var.get()
            gear_position = int(self.gear_position_var.get().split(":")[0])
            target_speed = float(self.target_speed_var.get())
            chassis_mode = int(self.chassis_mode_var.get().split(":")[0])
            target_angle = float(self.target_angle_var.get())
            
            # 验证参数
            if run_time <= 0:
                raise ValueError("运行时间必须为正数")
            if not 0 <= target_speed <= 10:
                raise ValueError("目标速度必须在0-10范围内")
            if not -180 <= target_angle <= 180:
                raise ValueError("目标角度必须在-180到180范围内")
            
            # 显示警告
            if target_speed > 5:
                if not messagebox.askyesno("警告", "目标速度较高，是否继续?"):
                    return
            
            if gear_position != 2:  # 如果不是N档
                if not messagebox.askyesno("警告", "档位不在N档，是否继续?"):
                    return
            
            # 禁用开始按钮，启用停止按钮
            self.start_button.state(['disabled'])
            self.stop_button.state(['!disabled'])
            self.running = True
            self.stop_requested = False
            
            # 在状态栏显示参数
            self.append_status("\n=== 开始控制 ===")
            self.append_status(f"运行时间: {run_time}秒")
            self.append_status(f"水泵开关: {'开' if water_pump else '关'}")
            self.append_status(f"远程使能: {'是' if remote_enable else '否'}")
            self.append_status(f"档位位置: {self.gear_position_var.get()}")
            self.append_status(f"目标速度: {target_speed}m/s")
            self.append_status(f"底盘模式: {self.chassis_mode_var.get()}")
            self.append_status(f"目标角度: {target_angle}度")
            
            # 在新线程中运行控制函数
            self.control_thread = threading.Thread(
                target=self.run_control,
                args=(run_time, water_pump, remote_enable, gear_position, 
                      target_speed, chassis_mode, target_angle),
                daemon=True
            )
            self.control_thread.start()
            
        except ValueError as e:
            messagebox.showerror("输入错误", str(e))
        except Exception as e:
            messagebox.showerror("错误", f"发生未知错误: {str(e)}")
    
    def run_control(self, run_time, water_pump, remote_enable, gear_position, 
                   target_speed, chassis_mode, target_angle):
        """实际运行控制函数"""
        try:
            # 调用实际的底盘控制函数
            self.append_status("\n发送控制信号...")
            
            # 这里调用您提供的 control_chassis 函数
            control_chassis(
                run_time=run_time,
                water_pump=water_pump,
                remote_enable=remote_enable,
                gear_position=gear_position,
                target_speed=target_speed,
                chassis_mode=chassis_mode,
                target_angle=target_angle
            )
            
            if self.stop_requested:
                self.append_status("\n已执行紧急停止!")
                # 发送停止信号
                self.append_status("发送停止信号...")
                self.append_status("档位: N档")
                self.append_status("目标速度: 0m/s")
                self.append_status("底盘模式: 驻车")
                self.append_status("目标角度: 0度")
            else:
                self.append_status("\n控制完成!")
            
        except Exception as e:
            self.append_status(f"\n控制过程中发生错误: {str(e)}")
        finally:
            # 恢复按钮状态
            self.running = False
            self.stop_requested = False
            self.root.after(0, lambda: self.start_button.state(['!disabled']))
            self.root.after(0, lambda: self.stop_button.state(['disabled']))
    
    def emergency_stop(self):
        """紧急停止"""
        if not self.running:
            return
        
        self.stop_requested = True
        self.append_status("\n用户请求紧急停止...")
        
        # 在紧急情况下，我们也可以直接调用 control_chassis 发送停止信号
        try:
            control_chassis(
                run_time=0.1,  # 短暂运行时间
                water_pump=False,
                remote_enable=False,
                gear_position=2,  # N档
                target_speed=0,
                chassis_mode=7,  # 驻车
                target_angle=0
            )
        except Exception as e:
            self.append_status(f"发送紧急停止信号时出错: {str(e)}")
    
    def append_status(self, message):
        """在状态栏添加消息"""
        self.status_text.configure(state=tk.NORMAL)
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.configure(state=tk.DISABLED)
        self.status_text.see(tk.END)
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = ChassisControlTester(root)
    app.run()