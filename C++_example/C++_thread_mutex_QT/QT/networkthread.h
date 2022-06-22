#pragma once
#define SKYNET_USERNAME     "peleng"
#define SKYNET_USERPASSWD   "1"

#define CAMERA_COUNT 2
#define IP_STREAMING_HD "0.0.0.0"
#define PORT_STREAMING_HD "5000"
#define MODEL_FOLDER "/home/peleng/PROJECT/skynet/third_party/trt_treacker/model/"
// 3264 x 2464
#define CAMERA_WIDTH 3264
#define CAMERA_HEIGHT 2464

//#define CAMERA_WIDTH 1920
//#define CAMERA_HEIGHT 1080
#define CONCAT 0  //vertocal 1 horizontal
#define CAMERA_MODULE_NAME       "[CAMERA]"
#define CAMERA_SCREENSHOT_NAME_TEMPLATE   "/tmp/outscreen_.png"


#define NETWORK_MODULE_NAME      "[NETWORK]"
#define MULTICAST_GROUP_IP       "224.1.1.10"
#define MULTICAST_PORT_HEARTBEAT "17778"
#define MULTICAST_PORT_COMMAND	 "17779"

#define DISPATCHER_MODULE_NAME      "[DISPATCHER]"
#define COMMAND_MAKE_SCREENSHOT          "screenshot"
#define COMMAND_SET                 "set"
#define PARAM_CAMERA                "camera"
#define PARAM_OFFSET_X              "offset_x"
#define PARAM_OFFSET_Y              "offset_y"


#include <QThread>
#include <QtNetwork/QUdpSocket>
#include <QNetworkInterface>
#include <QStringList>


class NetworkThread : public QThread
{
    Q_OBJECT
public:
    explicit NetworkThread(int iface_num = 0, QString* group = nullptr, QStringList* ports = nullptr);
    void run() override;
    void slotSendFile(QStringList file_name);
    void slotSendHeartBeat(QString message);
    void slotSendCommand(QString command);

private slots:
    void slotSendDatagramm(QUdpSocket* socket, int port, QByteArray message);
    void slotReceiveDatagramm();

private:
    QString multicast_group;
    QStringList multicast_ports;
    QUdpSocket* m_pudp_command;//экземпляр сокета
    QUdpSocket* m_pudp_heartbeat;//экземпляр сокета
    int current_iface;
    QString command;

signals:
    void signalCommand(QString command);
};
