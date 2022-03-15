var exec = require('cordova/exec');

function plugin() {

}

plugin.prototype.coolMethod = function(data) {
	data = '[{"part_no": "EHWIC-4ESG", "model": "Cisco Parts - Hardware", "brand": "Cisco"}, {"part_no": "HWIC-1FE", "model": "Cisco Parts - Hardware", "brand": "Cisco"}, {"part_no": "HWIC-1T", "model": "Cisco Parts - Hardware", "brand": "Cisco"}, {"part_no": "HWIC-2FE", "model": "Cisco Parts - Hardware", "brand": "Cisco"}, {"part_no": "HWIC-2T", "model": "Cisco Parts - Hardware", "brand": "Cisco"}, {"part_no": "HWIC-4ESW", "model": "Cisco Parts - Hardware", "brand": "Cisco"}, {"part_no": "NIM-ES2-4", "model": "Cisco Parts - Hardware", "brand": "Cisco"}, {"part_no": "VIC2-2E/M", "model": "Cisco Parts - Hardware", "brand": "Cisco"}, {"part_no": "VIC2-2FXO", "model": "Cisco Parts - Hardware", "brand": "Cisco"}, {"part_no": "VIC2-2FXS", "model": "Cisco Parts - Hardware", "brand": "Cisco"}, {"part_no": "WIC-1ENET", "model": "Cisco Parts - Hardware", "brand": "Cisco"}, {"part_no": "VWIC-1MFT-G703;VWIC-1MFT-T1/E1", "model": "Cisco Parts - Hardware", "brand": "Cisco"}, {"part_no": "VWIC2-1MFT-G703;VWIC2-1MFT-T1/E1", "model": "Cisco Parts - Hardware", "brand": "Cisco"}, {"part_no": "VWIC3-1MFT-G703;VWIC3-1MFT-T1/E1", "model": "Cisco Parts - Hardware", "brand": "Cisco"}, {"part_no": "VIC_2E/M", "model": "Cisco Parts - Hardware", "brand": "Cisco"}, {"part_no": "Huawei_MIC-1ELTE6-EA", "model": "Huawei Parts - Hardware", "brand": "Huawei"}]';
    exec(function(res){}, function(err){}, "cpcTFLitePlugin", "coolMethod", [data]);
}

module.exports = new plugin();