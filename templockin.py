def run(self, params):
    timeout = params['timeout']

    self.setup_lockin(params)
    time.sleep(0.01)

    loop_start = time.time()

    while (time.time() - loop_start) < timeout:
        self.capture_lockin()

    self.all_X = np.array(np.concatenate(self.lockin_X))
    self.all_Y = np.array(np.concatenate(self.lockin_Y))
    R = np.sqrt(self.all_X ** 2 + self.all_Y ** 2)
    Theta = np.arctan2(self.all_Y, self.all_X)

    # Time array
    t = np.arange(start=0, stop=len(self.all_X)/self.sample_rate, step=1/self.sample_rate)

    # FFT calculations
    iq = self.all_X + 1j*self.all_Y
    n_pts = len(iq)
    win = np.hanning(n_pts)
    IQwin = iq * win
    IQfft = np.fft.fftshift(np.fft.fft(IQwin))
    freqs_lock = np.fft.fftshift(np.fft.fftfreq(n_pts, 1.0 / self.sample_rate))
    psd_lock = (np.abs(IQfft) ** 2) / (self.sample_rate * np.sum(win ** 2))

    idx = np.argmax(psd_lock)
    print("Peak at", freqs_lock[idx], "Hz")

    # Create comprehensive plot with all lock-in outputs
    fig = plt.figure(figsize=(16, 10))
    
    # 1. FFT Spectrum
    ax1 = plt.subplot(3, 3, 1)
    ax1.semilogy(freqs_lock, psd_lock, label='Lock-in PSD')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Power (a.u.)')
    ax1.set_title('FFT Spectrum (baseband)')
    ax1.legend()
    ax1.grid(True)

    # 2. X vs Time
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(t, self.all_X, 'b-', linewidth=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('X (V)')
    ax2.set_title('In-phase (X) vs Time')
    ax2.grid(True)

    # 3. Y vs Time
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(t, self.all_Y, 'r-', linewidth=0.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Y (V)')
    ax3.set_title('Quadrature (Y) vs Time')
    ax3.grid(True)

    # 4. X vs Y (IQ plot)
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(self.all_X, self.all_Y, 'g.', markersize=1, alpha=0.5)
    ax4.set_xlabel('X (V)')
    ax4.set_ylabel('Y (V)')
    ax4.set_title('IQ Plot (X vs Y)')
    ax4.grid(True)
    ax4.axis('equal')

    # 5. R vs Time
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(t, R, 'm-', linewidth=0.5)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('R (V)')
    ax5.set_title('Magnitude (R) vs Time')
    ax5.grid(True)

    # 6. Theta vs Time
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(t, Theta, 'c-', linewidth=0.5)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Theta (rad)')
    ax6.set_title('Phase (Theta) vs Time')
    ax6.grid(True)

    # 7. All signals (X, Y, R, Theta) normalized
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(t, self.all_X/np.max(np.abs(self.all_X)), label='X (norm)', alpha=0.7)
    ax7.plot(t, self.all_Y/np.max(np.abs(self.all_Y)), label='Y (norm)', alpha=0.7)
    ax7.plot(t, R/np.max(R), label='R (norm)', alpha=0.7)
    ax7.plot(t, Theta/np.max(np.abs(Theta)), label='Theta (norm)', alpha=0.7)
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Normalized Amplitude')
    ax7.set_title('All Signals (Normalized)')
    ax7.legend()
    ax7.grid(True)

    # 8. R histogram
    ax8 = plt.subplot(3, 3, 8)
    ax8.hist(R, bins=50, edgecolor='black', alpha=0.7)
    ax8.set_xlabel('R (V)')
    ax8.set_ylabel('Count')
    ax8.set_title('Magnitude Distribution')
    ax8.grid(True)

    # 9. Theta histogram
    ax9 = plt.subplot(3, 3, 9)
    ax9.hist(Theta, bins=50, edgecolor='black', alpha=0.7)
    ax9.set_xlabel('Theta (rad)')
    ax9.set_ylabel('Count')
    ax9.set_title('Phase Distribution')
    ax9.grid(True)

    plt.tight_layout()

    if params['save_file']:
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        img_path = os.path.join(self.output_dir, f'lockin_results_rf_{self.ref_freq}.png')
        data = np.column_stack((R, Theta, self.all_X, self.all_Y))
        csv_path = os.path.join(self.output_dir, f'lockin_results_rf_{self.ref_freq}.csv')
        np.savetxt(csv_path, data, delimiter=",", header="R,Theta,X,Y", comments='', fmt='%.6f')
        plt.savefig(img_path, dpi=150)
    else:
        plt.show()
